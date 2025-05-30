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
from sklearn.cluster import KMeans
from Augmentation import Augmentation  # Import your original augmentation class

# ---------------------------
# Argument Parser
# ---------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='./models/source/best_model.pth')
parser.add_argument('--target_root', type=str, default='./repo-UGTST/data_preprocessed/PROMISE12')
parser.add_argument('--save_dir', type=str, default='./repo-UGTST/data_preprocessed/PROMISE12/slices_pseudo')
parser.add_argument('--select_ratio', type=float, default=0.05)
parser.add_argument('--tta_times', type=int, default=4)
parser.add_argument('--overwrite', action='store_true')
parser.add_argument('--seed', type=int, default=1337)
args = parser.parse_args()

# ---------------------------
# Split target cases
# ---------------------------
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

# ---------------------------
# Utility for Tensor Shape
# ---------------------------
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

# ---------------------------
# Entropy Computation
# ---------------------------
def softmax_entropy(pred):
    probs = torch.softmax(pred, dim=1)
    log_probs = torch.log(probs + 1e-6)
    entropy = -torch.sum(probs * log_probs, dim=1)
    return entropy

# ---------------------------
# TTA Wrapper using Augmentation
# ---------------------------
my_aug = Augmentation()  # Instantiate your existing augmentation

def apply_tta_transform(image):
    dummy_label = torch.zeros_like(image)
    sample = {'image': image.clone(), 'label': dummy_label.clone()}
    augmented = my_aug(sample)

    # Check if horizontal flip was applied by comparing difference
    if torch.allclose(augmented['image'], torch.flip(image, dims=[-1]), atol=1e-3):
        inverse_transform = lambda x: torch.flip(x, dims=[-1])
    else:
        inverse_transform = lambda x: x

    return augmented['image'], inverse_transform

# ---------------------------
# Global Aleatoric Uncertainty Aggregation (GAUA)
# ---------------------------
def compute_GAUA(entropy_map, num_bins=100):
    entropy_values = entropy_map.flatten().cpu().numpy()
    hist, bin_edges = np.histogram(entropy_values, bins=num_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    peaks = [i for i in range(1, len(hist) - 1) if hist[i] > hist[i - 1] and hist[i] >= hist[i + 1]]
    T_i = np.mean(entropy_values) if not peaks else min(bin_centers[peaks])

    mask = entropy_map > T_i
    U_i = entropy_map[mask].mean().item() if mask.sum() > 0 else entropy_map.mean().item()
    return U_i, T_i

# ---------------------------
# Custom Dataset for HDF5 slices
# ---------------------------
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

# ---------------------------
# Active Slice Selection
# ---------------------------
def select_uncertain_slices(model, dataloader, device, num_select=5, tta_times=4):
    model.eval()
    candidate_uncertainties, candidate_paths, candidate_features = [], [], []

    with torch.no_grad():
        for images, paths in tqdm(dataloader, desc="Uncertainty Estimation"):
            image = ensure_tensor4d(images).to(device)
            logits_list, features_list = [], []

            for _ in range(tta_times):
                aug_image, inv_transform = apply_tta_transform(image)
                output, feat = model(aug_image)
                output = inv_transform(output)
                logits_list.append(output.cpu())
                features_list.append(feat.cpu())

            avg_logits = torch.stack(logits_list).mean(0)
            entropy_map = softmax_entropy(avg_logits)
            U_i, T_i = compute_GAUA(entropy_map)
            candidate_uncertainties.append(U_i)
            candidate_paths.append(paths[0])
            avg_feature = torch.mean(torch.stack(features_list), dim=0)
            candidate_features.append(avg_feature.view(-1).cpu().numpy())

    candidate_capacity = 4 * num_select
    indices = np.argsort(candidate_uncertainties)[::-1][:candidate_capacity]
    Dtu_paths = [candidate_paths[i] for i in indices]
    Dtu_features = [candidate_features[i] for i in indices]

    features_array = np.stack(Dtu_features)
    kmeans = KMeans(n_clusters=num_select, init='k-means++').fit(features_array)
    selected = []
    for cluster_id in range(num_select):
        cluster_indices = np.where(kmeans.labels_ == cluster_id)[0]
        if len(cluster_indices) == 0:
            continue
        dists = np.linalg.norm(features_array[cluster_indices] - kmeans.cluster_centers_[cluster_id], axis=1)
        min_index = cluster_indices[np.argmin(dists)]
        selected.append(Dtu_paths[min_index])

    return selected

# ---------------------------
# Pseudo-label Generator
# ---------------------------
def generate_pseudo_labels(model, paths, save_dir, device, overwrite=False):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    with torch.no_grad():
        for path in tqdm(paths, desc="Generating Pseudo-labels"):
            save_path = os.path.join(save_dir, os.path.basename(path))
            if not overwrite and os.path.exists(save_path):
                continue
            with h5py.File(path, 'r') as f:
                image = f['image'][:]
            tensor = ensure_tensor4d(image).to(device)
            pred, _ = model(tensor)
            label = torch.argmax(torch.softmax(pred, dim=1), dim=1).squeeze(0).cpu().numpy()
            with h5py.File(save_path, 'w') as f:
                f.create_dataset('image', data=image.astype(np.float32))
                f.create_dataset('label', data=label.astype(np.uint8))

# ---------------------------
# Main Routine
# ---------------------------
def main(args):
    os.makedirs(args.save_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s',
                        handlers=[logging.FileHandler(os.path.join(args.save_dir, 'selection.log')),
                                  logging.StreamHandler()])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = UNet2D({'in_chns': 1, 'class_num': 2})
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)

    train_cases, val_cases, test_cases = split_target_cases(args.target_root, val_ratio=0.2, test_ratio=0.1, seed=args.seed)
    dataset = SliceDataset(args.target_root, case_filter=train_cases)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    select_num = max(1, int(len(dataset) * args.select_ratio))

    logging.info(f"Total training slices: {len(dataset)}, selecting top {select_num} uncertain slices")
    selected = select_uncertain_slices(model, dataloader, device, num_select=select_num, tta_times=args.tta_times)
    unselected = [p for p in dataset.h5_files if p not in selected]

    logging.info(f"Generating pseudo-labels for {len(unselected)} remaining slices")
    generate_pseudo_labels(model, unselected, args.save_dir, device, overwrite=args.overwrite)

    with open(os.path.join(args.save_dir, 'selected.txt'), 'w') as f:
        for path in selected:
            f.write(f"{os.path.basename(path)}\n")

    logging.info("Active sample selection and pseudo-labeling completed.")

if __name__ == '__main__':
    main(args)
