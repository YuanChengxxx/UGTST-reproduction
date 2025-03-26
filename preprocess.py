#!/usr/bin/env python3
import os
import glob
import h5py
import numpy as np
import SimpleITK as sitk
import random

# =========== 根据需要自定义 =========== #
data_dir = "/home/hpc/iwai/iwai119h/UGTST/repo-UGTST/data/PROMISE12"           # 仅此目录下查找文件
out_dir = "/home/hpc/iwai/iwai119h/UGTST/repo-UGTST/data_preprocessed/PROMISE12"
num_folds = 4
# ==================================== #

def min_max_normalize(volume: np.ndarray) -> np.ndarray:
    """对图像进行 min-max 归一化，映射到 [0,1]。"""
    v_min = volume.min()
    v_max = volume.max()
    if v_max - v_min > 1e-6:
        return (volume - v_min) / (v_max - v_min)
    return volume - v_min

def save_h5_volume(h5_path, image, label, spacing):
    """
    将整个 3D 图像、标签及 spacing 信息保存到一个 h5 文件中。
    - image: [slices, height, width], float32
    - label: [slices, height, width], uint8
    - spacing: (3,) 原始图像的 spacing
    """
    with h5py.File(h5_path, 'w') as f:
        f.create_dataset('image', data=image, compression="gzip")
        f.create_dataset('label', data=label, compression="gzip")
        f.create_dataset('spacing', data=spacing, compression="gzip")

def save_h5_slices(slices_dir, case_id, image, label):
    """
    将 3D 图像的每一层保存为独立的 h5 文件，文件名形如:
      slices_dir/CaseXX_slice{i}.h5
    """
    os.makedirs(slices_dir, exist_ok=True)
    num_slices = image.shape[0]
    for slice_ind in range(num_slices):
        h5_slice_path = os.path.join(slices_dir,
                                     f'{case_id}_slice{slice_ind}.h5')
        with h5py.File(h5_slice_path, 'w') as f:
            f.create_dataset('image', data=image[slice_ind], compression="gzip")
            f.create_dataset('label', data=label[slice_ind], compression="gzip")
    print(f"[INFO] {case_id} 的 {num_slices} 个切片已保存到 {slices_dir}")

def create_cv_splits(processed_cases, out_dir, k_folds=4):
    """
    将处理成功的病例随机打乱后分配到 k_folds 折，写入 folds.txt。
    文件格式:
        CaseXX   FoldX
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
    print(f"[INFO] 交叉验证划分已保存到 {folds_file}")

def process_case(img_file, seg_file, out_dir):
    """
    处理单个病例：
      1) 读取图像和标签
      2) min-max 归一化
      3) 将 3D 数据保存到 .h5
      4) 将每层切片另存为 .h5
    """
    base_name = os.path.basename(img_file).replace(".nii.gz", "")
    case_id = base_name  # 例如 "Case00"

    img_itk = sitk.ReadImage(img_file)
    seg_itk = sitk.ReadImage(seg_file)

    spacing = img_itk.GetSpacing()
    image = sitk.GetArrayFromImage(img_itk).astype(np.float32)
    label = sitk.GetArrayFromImage(seg_itk).astype(np.uint8)

    # 检查尺寸
    if image.shape != label.shape:
        print(f"[WARNING] 图像与标签尺寸不匹配: {img_file}")
        return ""

    # 归一化
    image = min_max_normalize(image)

    # 在 out_dir 下为当前病例创建目录
    os.makedirs(out_dir, exist_ok=True)
    case_dir = os.path.join(out_dir, case_id)
    os.makedirs(case_dir, exist_ok=True)

    # 保存 3D 数据
    main_h5_path = os.path.join(case_dir, f"{case_id}.h5")
    save_h5_volume(main_h5_path, image, label, spacing)

    # 保存切片
    slices_dir = os.path.join(case_dir, "slices")
    save_h5_slices(slices_dir, case_id, image, label)

    return case_id

def main():
    os.makedirs(out_dir, exist_ok=True)

    # 仅在 data_dir 目录下查找符合 "Case*.nii.gz" 的图像文件，排除 segmentation
    image_files = sorted(glob.glob(os.path.join(data_dir, "Case*.nii.gz")))
    image_files = [f for f in image_files if "_segmentation" not in f]

    processed_cases = []
    for img_file in image_files:
        seg_file = img_file.replace(".nii.gz", "_segmentation.nii.gz")
        if not os.path.exists(seg_file):
            print(f"[WARNING] {img_file} 对应的标签文件不存在：{seg_file}")
            continue

        base_name = os.path.basename(img_file)
        print("[INFO] Processing", base_name)
        case_id = process_case(img_file, seg_file, out_dir)
        if case_id:
            processed_cases.append(case_id)

    if processed_cases:
        create_cv_splits(processed_cases, out_dir, k_folds=num_folds)
    else:
        print("[ERROR] 没有处理成功的病例。")

    print("Preprocess completed.")

if __name__ == "__main__":
    main()
