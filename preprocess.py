
import os
import glob
import h5py
import numpy as np
import SimpleITK as sitk
import random


data_dir = "/home/hpc/iwai/iwai119h/UGTST/repo-UGTST/data/NCI-ISBI-2013"  # Manually modify the input path and output path to ensure that each preprocessed data is placed in the correct directory
out_dir = "/home/hpc/iwai/iwai119h/UGTST/repo-UGTST/data_preprocessed/NCI-ISBI-2013"
num_folds = 4


def min_max_normalize(volume: np.ndarray) -> np.ndarray:
   
    v_min = volume.min()
    v_max = volume.max()
    if v_max - v_min > 1e-6:
        return (volume - v_min) / (v_max - v_min)
    return volume - v_min

def save_h5_volume(h5_path, image, label, spacing):
   
    with h5py.File(h5_path, 'w') as f:
        f.create_dataset('image', data=image, compression="gzip")
        f.create_dataset('label', data=label, compression="gzip")
        f.create_dataset('spacing', data=spacing, compression="gzip")

def save_h5_slices(slices_dir, case_id, image, label):
    
    os.makedirs(slices_dir, exist_ok=True)
    num_slices = image.shape[0]
    for slice_ind in range(num_slices):
        h5_slice_path = os.path.join(slices_dir, f'{case_id}_slice{slice_ind}.h5')
        with h5py.File(h5_slice_path, 'w') as f:
            f.create_dataset('image', data=image[slice_ind], compression="gzip")
            f.create_dataset('label', data=label[slice_ind], compression="gzip")
    print(f"[INFO] {case_id} has {num_slices} slices saved to {slices_dir}")

def create_cv_splits(processed_cases, out_dir, k_folds=4):
   
    random.shuffle(processed_cases)
    folds = {i: [] for i in range(1, k_folds+1)}
    for idx, case_id in enumerate(processed_cases):
        fold_idx = (idx % k_folds) + 1
        folds[fold_idx].append(case_id)

    folds_file = os.path.join(out_dir, "folds.txt")
    with open(folds_file, 'w') as f:
        for fold_idx in range(1, k_folds+1):
            for case_id in folds[fold_idx]:
                f.write(f"{case_id}\tFold{fold_idx}\n")
    print(f"[INFO] The cross-validation split has been saved to {folds_file}")

def process_case(img_file, seg_file, out_dir):
    
    base_name = os.path.basename(img_file).replace(".nii.gz", "")
    case_id = base_name  # e.g. "Case00"

    img_itk = sitk.ReadImage(img_file)
    seg_itk = sitk.ReadImage(seg_file)

    spacing = img_itk.GetSpacing()
    image = sitk.GetArrayFromImage(img_itk).astype(np.float32)
    label = sitk.GetArrayFromImage(seg_itk).astype(np.uint8)

    # Check the dimensions
    if image.shape != label.shape:
        print(f"[WARNING] Image and label dimensions do not match: {img_file}")
        return ""

    # Perform min-max normalization
    image = min_max_normalize(image)

    # Create a directory for this case under out_dir
    os.makedirs(out_dir, exist_ok=True)
    case_dir = os.path.join(out_dir, case_id)
    os.makedirs(case_dir, exist_ok=True)

    # Save the 3D volume
    main_h5_path = os.path.join(case_dir, f"{case_id}.h5")
    save_h5_volume(main_h5_path, image, label, spacing)

    # Save slices
    slices_dir = os.path.join(case_dir, "slices")
    save_h5_slices(slices_dir, case_id, image, label)

    return case_id

def main():
    os.makedirs(out_dir, exist_ok=True)

    # Search for "Case*.nii.gz" in data_dir, excluding those containing "_segmentation"
    image_files = sorted(glob.glob(os.path.join(data_dir, "Case*.nii.gz")))
    image_files = [f for f in image_files if "_segmentation" not in f]

    processed_cases = []
    for img_file in image_files:
        seg_file = img_file.replace(".nii.gz", "_segmentation.nii.gz")
        if not os.path.exists(seg_file):
            print(f"[WARNING] The label file for {img_file} does not exist: {seg_file}")
            continue

        base_name = os.path.basename(img_file)
        print("[INFO] Processing", base_name)
        case_id = process_case(img_file, seg_file, out_dir)
        if case_id:
            processed_cases.append(case_id)

    if processed_cases:
        create_cv_splits(processed_cases, out_dir, k_folds=num_folds)
    else:
        print("[ERROR] No cases were successfully processed.")

    print("Preprocess completed.")

if __name__ == "__main__":
    main()
