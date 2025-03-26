import os
import numpy as np
import torch
import h5py
from glob import glob
from tqdm import tqdm
import torch.nn.functional as F
from medpy.metric.binary import hd95


def evaluate_model_on_volumes(model, data_root, device, case_filter=None):
    """
    Evaluate a 2D segmentation model on 3D volumes stored as slices under PROMISE12/CaseXX/slices/*.h5.
    Ground truth is loaded from PROMISE12/CaseXX/CaseXX.h5.

    :param model: trained 2D segmentation model
    :param data_root: root directory of PROMISE12 dataset
    :param device: torch.device ("cuda" or "cpu")
    :param case_list: optional list of case names (e.g., ["Case00", "Case01"])
    :return: average dice, average hd95
    """

    model.eval()
    dice_scores = []
    hd95_scores = []

    
    all_case_dirs = sorted(glob(os.path.join(data_root, "Case*")))
    if case_filter is not None:
        case_dirs = [os.path.join(data_root, c) for c in case_filter if os.path.isdir(os.path.join(data_root, c))]
    else:
        case_dirs = all_case_dirs

    with torch.no_grad():
        for case in tqdm(case_dirs, desc="Evaluating Volumes"):
            case_id = os.path.basename(case)
            slice_paths = sorted(glob(os.path.join(case, "slices", "*.h5")))
            pred_slices = []

            for sp in slice_paths:
                with h5py.File(sp, 'r') as f:
                    image = f['image'][:]
                if image.ndim == 2:
                    image = image[np.newaxis, :, :]
                tensor = torch.tensor(image[np.newaxis], dtype=torch.float32).to(device)  # [1, 1, H, W]
                pred_logit, _ = model(tensor)
                if isinstance(pred_logit, (list, tuple)):
                    pred_logit = pred_logit[0]  
                pred_mask = torch.argmax(F.softmax(pred_logit, dim=1), dim=1).squeeze(0).cpu().numpy()  # [H, W]
                pred_slices.append(pred_mask)

            pred_volume = np.stack(pred_slices, axis=0)  # [D, H, W]

           
            gt_path = os.path.join(case, f"{case_id}.h5")
            with h5py.File(gt_path, 'r') as f:
                gt_volume = f['label'][:]

            pred = pred_volume.astype(np.uint8)
            gt = gt_volume.astype(np.uint8)
            # print(f"[Eval] {case_id} Prediction unique values:", np.unique(pred))
            if pred.any() and gt.any():
                dice = 2.0 * np.sum((pred == 1) & (gt == 1)) / (np.sum(pred == 1) + np.sum(gt == 1) + 1e-5)
                hd = hd95(pred, gt)
                hd95_scores.append(hd)
            else:
                dice = 1.0 if np.array_equal(pred, gt) else 0.0
            dice_scores.append(dice)

    return np.mean(dice_scores), np.mean(hd95_scores) if hd95_scores else np.nan
