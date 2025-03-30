import os
import glob
import h5py
import numpy as np
import SimpleITK as sitk
import random

# Define input and output directories
data_dir = "./repo-UGTST/data/PROMISE12"           # Path to the raw MRI dataset
out_dir = "./repo-UGTST/data_preprocessed/PROMISE12"  # Output path for preprocessed data
num_folds = 4  # Number of folds to create for cross-validation

def min_max_normalize(volume: np.ndarray) -> np.ndarray:
    """
    Normalize the input volume to the range [0, 1] using min-max normalization.
    If the volume has no variation, subtract the min value only.
    """
    v_min = volume.min()
    v_max = volume.max()
    if v_max - v_min > 1e-6:
        return (volume - v_min) / (v_max - v_min)
    return volume - v_min  # Handle near-constant volumes

def save_h5_volume(h5_path, image, label, spacing):
    """
    Save the entire 3D image volume and label as an HDF5 (.h5) file.
    Also save spacing information to preserve resolution metadata.
    """
    with h5py.File(h5_path, 'w') as f:
        f.create_dataset('image', data=image, compression="gzip")
        f.create_dataset('label', data=label, compression="gzip")
        f.create_dataset('spacing', data=spacing, compression="gzip")

def save_h5_slices(slices_dir, case_id, image, label):
    """
    Save each 2D slice from the 3D image and label volumes as separate HDF5 files.
    Useful for slice-based training using 2D networks (e.g. 2D U-Net).
    """
    os.makedirs(slices_dir, exist_ok=True)
    num_slices = image.shape[0]
    for slice_ind in range(num_slices):
        h5_slice_path = os.path.join(slices_dir, f'{case_id}_slice{slice_ind}.h5')
        with h5py.File(h5_slice_path, 'w') as f:
            f.create_dataset('image', data=image[slice_ind], compression="gzip")
            f.create_dataset('label', data=label[slice_ind], compression="gzip")

def create_cv_splits(processed_cases, out_dir, k_folds=4):
    """
    Randomly assign the list of processed case IDs into k-folds for cross-validation.
    Save the assignment to a text file with format: <case_id> <fold_number>
    """
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

def process_case(img_file, seg_file, out_dir):
    """
    Process a single MRI case:
    - Read the image and corresponding segmentation
    - Normalize the image volume
    - Save the 3D volume and each 2D slice to HDF5 format
    """
    base_name = os.path.basename(img_file).replace(".nii.gz", "")
    case_id = base_name

    # Load image and label using SimpleITK
    img_itk = sitk.ReadImage(img_file)
    seg_itk = sitk.ReadImage(seg_file)

    spacing = img_itk.GetSpacing()
    image = sitk.GetArrayFromImage(img_itk).astype(np.float32)  # Shape: [slices, height, width]
    label = sitk.GetArrayFromImage(seg_itk).astype(np.uint8)

    # Normalize the image
    image = min_max_normalize(image)

    # Create case-specific output directory
    os.makedirs(out_dir, exist_ok=True)
    case_dir = os.path.join(out_dir, case_id)
    os.makedirs(case_dir, exist_ok=True)

    # Save the full 3D volume
    main_h5_path = os.path.join(case_dir, f"{case_id}.h5")
    save_h5_volume(main_h5_path, image, label, spacing)

    # Save individual 2D slices
    slices_dir = os.path.join(case_dir, "slices")
    save_h5_slices(slices_dir, case_id, image, label)

    return case_id

def main():
    """
    Main function to:
    - Iterate over all MRI files
    - Preprocess each case
    - Generate k-fold split file for cross-validation
    """
    os.makedirs(out_dir, exist_ok=True)

    # Find all Case*.nii.gz image files, excluding label files
    image_files = sorted(glob.glob(os.path.join(data_dir, "Case*.nii.gz")))
    image_files = [f for f in image_files if "_segmentation" not in f]

    processed_cases = []
    for img_file in image_files:
        # Derive the corresponding segmentation file name
        seg_file = img_file.replace(".nii.gz", "_segmentation.nii.gz")
        base_name = os.path.basename(img_file)
        print("[INFO] Processing", base_name)
        case_id = process_case(img_file, seg_file, out_dir)
        if case_id:
            processed_cases.append(case_id)

    # Create fold splits for cross-validation
    if processed_cases:
        create_cv_splits(processed_cases, out_dir, k_folds=num_folds)
    else:
        print("[ERROR] There were no successful cases.")

    print("Preprocess completed.")

if __name__ == "__main__":
    main()
