import os
import argparse
import logging
import h5py
import numpy as np
from glob import glob
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import SGD
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import random

from Unet2D import UNet2D
from Augmentation import Augmentation
from evaluator import evaluate_model_on_volumes
from losses import DiceLoss, FocalLoss, ComboLoss

parser = argparse.ArgumentParser()
parser.add_argument('--data_root', type=str,
                    default="./repo-UGTST/data_preprocessed/NCI-ISBI-2013")
parser.add_argument('--fold_file', type=str,
                    default="./repo-UGTST/data_preprocessed/NCI-ISBI-2013/folds.txt")
parser.add_argument('--val_fold', type=int, default=1)
parser.add_argument('--epochs', type=int, default=400)
parser.add_argument('--batch_size', type=int, default=24)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--save_dir', type=str, default='./models/source')
parser.add_argument('--seed', type=int, default=1337)
args = parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def generate_folds_txt(data_root, fold_file_path, num_folds=4, seed=1337):
    case_dirs = sorted([d for d in os.listdir(data_root) if d.startswith('Case')])
    random.seed(seed)
    random.shuffle(case_dirs)
    folds = [[] for _ in range(num_folds)]
    for i, case in enumerate(case_dirs):
        folds[i % num_folds].append(case)
    with open(fold_file_path, 'w') as f:
        for i in range(num_folds):
            f.write(f"Fold{i}: {' '.join(folds[i])}\n")
    print(f"[Info] Generated {fold_file_path} with {num_folds} folds.")

def parse_folds(fold_file):
    fold_dict = {}
    with open(fold_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or ':' not in line:
                continue
            parts = line.split(":")
            if len(parts) != 2:
                continue
            fold_name, cases_str = parts
            fold_id = int(fold_name.strip().replace("Fold", ""))
            cases = cases_str.strip().split()
            fold_dict[fold_id] = cases
    return fold_dict

class ProstateDataset(Dataset):
    def __init__(self, preprocessed_dir, case_ids, transform=None):
        self.data_paths = []
        for case_id in case_ids:
            self.data_paths.extend(sorted(glob(os.path.join(preprocessed_dir, case_id, "slices", "*.h5"))))
        self.transform = transform

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        with h5py.File(self.data_paths[idx], 'r') as f:
            image = f['image'][:]
            label = f['label'][:]
        image = torch.tensor(image[np.newaxis, ...], dtype=torch.float32)
        label = torch.tensor(label[np.newaxis, ...], dtype=torch.long)
        label = torch.where(label > 0, torch.tensor(1, dtype=torch.long), torch.tensor(0, dtype=torch.long))
        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        return sample

def adjust_learning_rate(optimizer, base_lr, iter, max_iter, power=0.9):
    lr = base_lr * ((1 - float(iter) / max_iter) ** power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def train(args):
    os.makedirs(args.save_dir, exist_ok=True)
    logging.basicConfig(filename=os.path.join(args.save_dir, 'train.log'), level=logging.INFO, format='%(asctime)s %(message)s')
    console = logging.StreamHandler()
    logging.getLogger().addHandler(console)

    writer = SummaryWriter(log_dir=os.path.join(args.save_dir, 'tensorboard'))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(args.fold_file):
        generate_folds_txt(args.data_root, args.fold_file)

    fold_dict = parse_folds(args.fold_file)
    val_ids = fold_dict[args.val_fold]
    train_ids = [case for fid, cases in fold_dict.items() if fid != args.val_fold for case in cases]

    logging.info(f"Train cases ({len(train_ids)}): {train_ids}")
    logging.info(f"Val cases ({len(val_ids)}): {val_ids}")

    train_set = ProstateDataset(args.data_root, train_ids, transform=Augmentation())
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4)

    model = UNet2D({'in_chns': 1, 'class_num': 2, 'multiscale_pred': True}).to(device)
    optimizer = SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    max_iter = args.epochs * len(train_loader)

    global_step = 0
    best_dice = 0.0
    model.train()
    loss_fn = ComboLoss(first=DiceLoss(), second=nn.CrossEntropyLoss(), weight=0.3)  

    for epoch in range(args.epochs):
        loop = tqdm(train_loader, total=len(train_loader), desc=f"Epoch [{epoch+1}/{args.epochs}]")
        for batch in loop:
            model.train()
            images = batch['image'].to(device)
            labels = batch['label'].to(device)

            outputs, _ = model(images)
            if isinstance(outputs, (list, tuple)):
                loss = sum([loss_fn(out, labels) for out in outputs]) / len(outputs)
            else:
                loss = loss_fn(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            current_lr = adjust_learning_rate(optimizer, args.lr, global_step, max_iter)
            global_step += 1

            loop.set_postfix(loss=loss.item(), lr=current_lr)
            writer.add_scalar('train/loss', loss.item(), global_step)
            writer.add_scalar('train/lr', current_lr, global_step)

        val_dice, val_hd = evaluate_model_on_volumes(model, args.data_root, device, case_filter=val_ids)
        writer.add_scalar('val/dice', val_dice, epoch)
        writer.add_scalar('val/hd95', val_hd, epoch)
        logging.info(f"Epoch {epoch+1}, Val Dice: {val_dice:.4f}, Val HD95: {val_hd:.4f}")

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'best_model.pth'))
            logging.info(f">>> New best model saved with Dice {val_dice:.4f}")

        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), os.path.join(args.save_dir, f'model_epoch_{epoch+1}.pth'))

    writer.close()

if __name__ == '__main__':
    set_seed(args.seed)
    train(args)
