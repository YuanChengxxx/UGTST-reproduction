import os
import argparse
import h5py
import torch
import numpy as np
from tqdm import tqdm
from glob import glob
from medpy.metric import binary
from Unet2D import UNet2D

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str,
                    default='./repo-UGTST/outputs/target_training/Stage2_best_model.pth')
parser.add_argument('--data_root', type=str,
                    default='./repo-UGTST/data_preprocessed/PROMISE12')
parser.add_argument('--test_list', type=str,
                    default='./repo-UGTST/data_preprocessed/PROMISE12/target_test.txt')
args = parser.parse_args()

def load_case_list(txt_path):
    with open(txt_path, 'r') as f:
        return [line.strip() for line in f if line.strip()]


def evaluate_single_case(model, case_dir, device):
    slice_paths = sorted(glob(os.path.join(case_dir, "slices", "*.h5")))
    preds = []
    for sp in slice_paths:
        with h5py.File(sp, 'r') as f:
            image = f['image'][:]
            label = f['label'][:]

        input_tensor = torch.tensor(image[np.newaxis, np.newaxis], dtype=torch.float32).to(device)
        model.eval()
        with torch.no_grad():
            pred_logits, _ = model(input_tensor)
            pred = torch.argmax(torch.softmax(pred_logits, dim=1), dim=1).squeeze(0).cpu().numpy()
        preds.append(pred)

    pred_volume = np.stack(preds, axis=0)

    gt_path = os.path.join(case_dir, os.path.basename(case_dir) + ".h5")
    with h5py.File(gt_path, 'r') as f:
        gt_volume = f['label'][:]

    pred = pred_volume.astype(np.uint8)
    gt = gt_volume.astype(np.uint8)

    if pred.any() and gt.any():
        dice = 2.0 * np.sum((pred == 1) & (gt == 1)) / (np.sum(pred == 1) + np.sum(gt == 1) + 1e-5)
        hd95 = binary.hd95(pred, gt)
    else:
        dice = 1.0 if np.array_equal(pred, gt) else 0.0
        hd95 = 0.0 if dice == 1.0 else np.nan
    return dice, hd95


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet2D({'in_chns': 1, 'class_num': 2}).to(device)

    model.load_state_dict(torch.load(args.model_path, map_location=device))

    test_cases = load_case_list(args.test_list)
    all_dice, all_hd95 = [], []

    for case in tqdm(test_cases, desc="Evaluating on Test Set"):
        case_dir = os.path.join(args.data_root, case)
        dice, hd = evaluate_single_case(model, case_dir, device)
        all_dice.append(dice)
        all_hd95.append(hd)
        print(f"{case}: Dice={dice:.4f}, HD95={hd:.2f}")

    dice_mean = np.mean(all_dice)
    dice_std = np.std(all_dice)
    hd95_mean = np.nanmean(all_hd95)
    hd95_std = np.nanstd(all_hd95)

    print("\n====== Test Set Performance ======")
    print(f"Dice  : {dice_mean:.3f} ± {dice_std:.3f}")
    print(f"HD95  : {hd95_mean:.2f} ± {hd95_std:.2f}")


if __name__ == '__main__':

    main(args)
