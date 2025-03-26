import os
import h5py
import logging
import torch
import argparse
import numpy as np
from glob import glob
from tqdm import tqdm
import random
from Unet2D import UNet2D
from torch.utils.data import Dataset, DataLoader

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='./models/source/best_model.pth')
parser.add_argument('--target_root', type=str,
                    default='./repo-UGTST/data_preprocessed/PROMISE12')
parser.add_argument('--save_dir', type=str,
                    default='./repo-UGTST/data_preprocessed/PROMISE12/slices_pseudo')
parser.add_argument('--select_ratio', type=float, default=0.05)
parser.add_argument('--tta_times', type=int, default=4)
parser.add_argument('--overwrite', action='store_true')
parser.add_argument('--seed', type=int, default=1337)
args = parser.parse_args()

def split_target_cases(root, val_ratio=0.2, test_ratio=0.1, seed=1337):
    case_dirs = sorted([d for d in os.listdir(root) if d.startswith("Case")])
    random.seed(seed)
    random.shuffle(case_dirs)

    total = len(case_dirs)
    test_count = int(total * test_ratio)
    val_count = int(total * val_ratio)
    train_count = total - val_count - test_count

    test_cases = case_dirs[:test_count]
    val_cases = case_dirs[test_count:test_count + val_count]
    train_cases = case_dirs[test_count + val_count:]

    with open(os.path.join(root, "target_train.txt"), 'w') as f:
        f.writelines([c + '\n' for c in train_cases])
    with open(os.path.join(root, "target_val.txt"), 'w') as f:
        f.writelines([c + '\n' for c in val_cases])
    with open(os.path.join(root, "target_test.txt"), 'w') as f:
        f.writelines([c + '\n' for c in test_cases])

    return train_cases, val_cases, test_cases


def ensure_tensor4d(tensor):
    if isinstance(tensor, np.ndarray):
        tensor = torch.tensor(tensor, dtype=torch.float32)
    if tensor.dim() == 4:
        return tensor
    elif tensor.dim() == 3:
        return tensor.unsqueeze(0)
    elif tensor.dim() == 2:
        return tensor.unsqueeze(0).unsqueeze(0)
    else:
        raise ValueError(f"Unsupported tensor shape: {tensor.shape}")

def softmax_entropy(pred):
    probs = torch.softmax(pred, dim=1)
    log_probs = torch.log(probs + 1e-6)
    entropy = -torch.sum(probs * log_probs, dim=1)
    return entropy

def aggregate_uncertainty(entropy_map):
    return entropy_map.mean().item()

class SliceDataset(Dataset):
    def __init__(self, root, case_filter=None):
        all_files = sorted(glob(os.path.join(root, "Case*/slices/*.h5")))
        if case_filter:
            self.h5_files = [p for p in all_files if any(f"/{c}/" in p for c in case_filter)]
        else:
            self.h5_files = all_files

    def __len__(self):
        return len(self.h5_files)

    def __getitem__(self, idx):
        path = self.h5_files[idx]
        with h5py.File(path, 'r') as f:
            image = f['image'][:]
        image = torch.tensor(image, dtype=torch.float32)
        if image.dim() == 2:
            image = image.unsqueeze(0)
        return image, path

def select_uncertain_slices(model, dataloader, device, num_select=5, tta_times=4):
    model.eval()
    uncertainties = []
    with torch.no_grad():
        for images, paths in tqdm(dataloader, desc="Uncertainty Estimation"):
            image = ensure_tensor4d(images).to(device)
            all_logits = []
            for _ in range(tta_times):
                noise = torch.clamp(torch.randn_like(image) * 0.05, -0.1, 0.1)
                noisy_input = image + noise
                output, _ = model(noisy_input)
                all_logits.append(output.cpu())
            avg_logits = torch.stack(all_logits).mean(0)
            entropy_map = softmax_entropy(avg_logits)
            avg_uncertainty = aggregate_uncertainty(entropy_map)
            uncertainties.append((avg_uncertainty, paths[0]))
    uncertainties.sort(reverse=True)
    selected = [p for _, p in uncertainties[:num_select]]
    return selected

def generate_pseudo_labels(model, paths, save_dir, device, overwrite=False):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    with torch.no_grad():
        for path in tqdm(paths, desc="Generating Pseudo-labels"):
            filename = os.path.basename(path)
            save_path = os.path.join(save_dir, filename)
            if not overwrite and os.path.exists(save_path):
                continue
            with h5py.File(path, 'r') as f:
                image = f['image'][:]
            slice_tensor = ensure_tensor4d(image).to(device)
            pred, _ = model(slice_tensor)
            pred_label = torch.argmax(torch.softmax(pred, dim=1), dim=1).squeeze(0).cpu().numpy()
            with h5py.File(save_path, 'w') as f:
                f.create_dataset('image', data=image.astype(np.float32))
                f.create_dataset('label', data=pred_label.astype(np.uint8))

def main(args):
    os.makedirs(args.save_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s',
                        handlers=[logging.FileHandler(os.path.join(args.save_dir, 'selection.log')), logging.StreamHandler()])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet2D({'in_chns': 1, 'class_num': 2})
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)

    train_cases, val_cases, test_cases = split_target_cases(args.target_root, val_ratio=0.2, test_ratio=0.1, seed=args.seed)

    dataset = SliceDataset(args.target_root, case_filter=train_cases)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    total = len(dataset)
    select_num = max(1, int(total * args.select_ratio))
    logging.info(f"Total training slices: {total}, selecting top {select_num} uncertain slices")

    selected = select_uncertain_slices(model, dataloader, device, num_select=select_num, tta_times=args.tta_times)
    unselected = [p for p in dataset.h5_files if p not in selected]
    logging.info(f"Generating pseudo-labels for {len(unselected)} remaining slices")

    generate_pseudo_labels(model, unselected, args.save_dir, device, overwrite=args.overwrite)

    selected_txt_path = os.path.join(args.save_dir, 'selected.txt')
    with open(selected_txt_path, 'w') as f:
        for path in selected:
            f.write(f"{os.path.basename(path)}\n")

    logging.info("Active sample selection and pseudo-labeling completed.")

if __name__ == '__main__':

    main(args)
