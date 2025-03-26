
# 🧬 UGTST-Reproduction

This repository reproduces the experiments from the paper:

> **"An Uncertainty-guided Tiered Self-training Framework for Active Source-free Domain Adaptation in Prostate Segmentation" (MICCAI 2023)**

This repo supports:
- ✅ Source domain training
- ✅ Active sample selection
- ✅ Target domain self-training (Stage 1 + Stage 2)  
on the **PROMISE12** dataset.

---

## 📁 Project Structure

```
UGTST-reproduction/
├── data_preprocessed/       # Preprocessed PROMISE12 & NCI-ISBI-2013 data (not uploaded)
│   └── PROMISE12/
│   └── NCI-ISBI-2013/
├── models/                  # Trained model checkpoints
├── scripts/                 # All training, selection, evaluation scripts
├── outputs/                 # Logs and saved models
├── configs/                 # (Optional) Config files
├── preprocess.py            # Data preprocessing
├── train_source.py          # Source domain training
├── train_target.py          # Target domain self-training
├── active_select.py         # Active selection & pseudo-labeling
├── test.py                  # Evaluation script
├── requirements.txt
└── README.md
```

---

## 📦 Installation

```bash
git clone https://github.com/YOUR_NAME/UGTST-reproduction.git
cd UGTST-reproduction

conda create -n ugtst python=3.10
conda activate ugtst
pip install -r requirements.txt
```

---

## 📂 Data Preparation

1. Download the **PROMISE12** dataset from  
👉 https://promise12.grand-challenge.org/
2. Download the **NCI-ISBI 2013** dataset from  
👉 [https://promise12.grand-challenge.org/](https://www.cancerimagingarchive.net/analysis-result/isbi-mr-prostate-2013/)

3. Place raw `.nii.gz` files in `./data/PROMISE12/` and `./data/NCI-ISBI-2013/`

4. Modify `preprocess.py`:
```python
data_dir = "./data/PROMISE12/"
out_dir = "./data_preprocessed/PROMISE12"
# Similarly for NCI-ISBI-2013
```

4. Run preprocessing:
```bash
python preprocess.py
```

This generates:
```
data_preprocessed/PROMISE12/
├── CaseXX/
│   ├── CaseXX.h5            # 3D volume with label
│   └── slices/*.h5          # 2D slices
├── slices_pseudo/           # Generated in next step
├── target_train.txt
├── target_val.txt
├── target_test.txt
└── folds.txt
```

---

## 🚀 Training Pipeline

### 🔹 1. Train on Source Domain

```bash
python train_source.py   --data_root ./data_preprocessed/NCI-ISBI-2013   --fold_file ./data_preprocessed/NCI-ISBI-2013/folds.txt   --save_dir ./models/source
```

---

### 🔹 2. Active Sample Selection + Pseudo Label Generation

```bash
python active_select.py   --model_path ./models/source/best_model.pth   --target_root ./data_preprocessed/PROMISE12   --save_dir ./data_preprocessed/PROMISE12/slices_pseudo   --select_ratio 0.05
```

This will:
- ✅ Create `selected.txt` for most uncertain slices
- ✅ Generate pseudo labels for the rest

---

### 🔹 3. Target Domain Self-training

```bash
python train_target.py   --model_path ./models/source/best_model.pth   --data_root ./data_preprocessed/PROMISE12   --pseudo_dir ./data_preprocessed/PROMISE12/slices_pseudo   --active_txt ./data_preprocessed/PROMISE12/slices_pseudo/selected.txt   --train_cases ./data_preprocessed/PROMISE12/target_train.txt   --val_cases ./data_preprocessed/PROMISE12/target_val.txt   --save_dir ./outputs/target_training   --epochs_stage1 50   --epochs_stage2 50
```

---

## 🧪 Testing & Evaluation

```bash
python test.py   --model_path ./outputs/target_training/Stage2_best_model.pth   --data_root ./data_preprocessed/PROMISE12   --test_list ./data_preprocessed/PROMISE12/target_test.txt
```

You will get:
- ✅ Dice coefficient & HD95 on the test set
- ✅ Per-case metrics printed in console

---


