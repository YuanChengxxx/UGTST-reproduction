import os
import argparse
import logging
import numpy as np
import h5py
from glob import glob
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import random

from Unet2D import UNet2D
from evaluator import evaluate_model_on_volumes

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default="./repo-UGTST/repo1/models/source/best_model.pth")
parser.add_argument('--data_root', type=str, default="./repo-UGTST/data_preprocessed/PROMISE12")
parser.add_argument('--save_dir', type=str, default="./repo-UGTST/outputs/target_training")
parser.add_argument('--pseudo_dir', type=str, default="./repo-UGTST/data_preprocessed/PROMISE12/slices_pseudo")
parser.add_argument('--active_txt', type=str, default="./repo-UGTST/data_preprocessed/PROMISE12/slices_pseudo/selected.txt")
parser.add_argument('--val_cases', type=str, default="./repo-UGTST/data_preprocessed/PROMISE12/target_val.txt")
parser.add_argument('--train_cases', type=str, default="./repo-UGTST/data_preprocessed/PROMISE12/target_train.txt")
parser.add_argument('--epochs_stage1', type=int, default=50)
parser.add_argument('--epochs_stage2', type=int, default=50)
parser.add_argument('--batch_size', type=int, default=24)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--seed', type=int, default=1337)
args = parser.parse_args()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def dice_loss(pred, target, smooth=1e-5):
    pred = torch.softmax(pred, dim=1)
    target_onehot = torch.nn.functional.one_hot(target.squeeze(1), num_classes=pred.shape[1]).permute(0, 3, 1, 2).float()
    intersect = torch.sum(pred * target_onehot)
    union = torch.sum(pred + target_onehot)
    return 1 - (2 * intersect + smooth) / (union + smooth)

def load_case_list(txt_path):
    with open(txt_path, 'r') as f:
        return [line.strip() for line in f if line.strip()]

def train_stage(model, train_loader, val_root, val_case_list, device, optimizer, epochs, stage_name, log_dir):
    best_dice = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        loop = tqdm(train_loader, total=len(train_loader), desc=f"{stage_name} Epoch {epoch}")
        for batch in loop:
            img = batch['image'].to(device)
            lbl = batch['label'].to(device)
            out, _ = model(img)
            loss = 0.5 * nn.CrossEntropyLoss()(out, lbl.squeeze(1)) + 0.5 * dice_loss(out, lbl)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loop.set_postfix(loss=loss.item())

        dice, hd = evaluate_model_on_volumes(model, val_root, device, case_filter=val_case_list)
        logging.info(f"{stage_name} Epoch {epoch} | Val Dice: {dice:.4f} | Val HD95: {hd:.4f}")
        if dice > best_dice:
            best_dice = dice
            torch.save(model.state_dict(), os.path.join(log_dir, f"{stage_name}_best_model.pth"))

class GTOnlyDataset(Dataset):
    def __init__(self, root_dir, selected_txt, case_filter=None):
        with open(selected_txt, 'r') as f:
            selected_names = set([line.strip() for line in f])

        all_paths = sorted(glob(os.path.join(root_dir, "Case*/slices/*.h5")))
        if case_filter:
            all_paths = [p for p in all_paths if any(f"/{c}/" in p for c in case_filter)]

        self.paths = [p for p in all_paths if os.path.basename(p) in selected_names]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        h5_path = self.paths[idx]
        with h5py.File(h5_path, 'r') as f:
            image = f['image'][:]
            label = f['label'][:]
        image = torch.tensor(image[np.newaxis], dtype=torch.float32)
        label = torch.tensor(label[np.newaxis], dtype=torch.long)
        return {'image': image, 'label': label}

class PseudoOnlyDataset(Dataset):
    def __init__(self, pseudo_dir):
        self.paths = sorted(glob(os.path.join(pseudo_dir, "*.h5")))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        h5_path = self.paths[idx]
        with h5py.File(h5_path, 'r') as f:
            image = f['image'][:]
            label = f['label'][:]
        image = torch.tensor(image[np.newaxis], dtype=torch.float32)
        label = torch.tensor(label[np.newaxis], dtype=torch.long)
        return {'image': image, 'label': label}

if __name__ == '__main__':
    os.makedirs(args.save_dir, exist_ok=True)
    logging.basicConfig(filename=os.path.join(args.save_dir, 'train.log'),
                        level=logging.INFO,
                        format='%(asctime)s %(message)s')
    console = logging.StreamHandler()
    logging.getLogger().addHandler(console)

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet2D({'in_chns': 1, 'class_num': 2}).to(device)
    model.load_state_dict(torch.load(args.model_path))
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)

    train_cases = load_case_list(args.train_cases)
    val_cases = load_case_list(args.val_cases)

    if args.epochs_stage1 > 0:
        logging.info(">>> Starting Stage 1 Training")
        train_dataset1 = GTOnlyDataset(args.data_root, args.active_txt, case_filter=train_cases)
        train_loader1 = DataLoader(train_dataset1, batch_size=args.batch_size, shuffle=True, num_workers=4)
        train_stage(model, train_loader1, args.data_root, val_cases, device, optimizer, args.epochs_stage1, "Stage1", args.save_dir)

    logging.info(">>> Starting Stage 2 Training")
    model.load_state_dict(torch.load(os.path.join(args.save_dir, 'Stage1_best_model.pth')))
    train_dataset2 = PseudoOnlyDataset(args.pseudo_dir)
    train_loader2 = DataLoader(train_dataset2, batch_size=args.batch_size, shuffle=True, num_workers=4)
    train_stage(model, train_loader2, args.data_root, val_cases, device, optimizer, args.epochs_stage2, "Stage2", args.save_dir)

    logging.info(">>> Training Finished")
