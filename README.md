# UGTST-Reproduction

This repository reproduces the experiments from the paper:

**"An Uncertainty-guided Tiered Self-training Framework for Active Source-free Domain Adaptation in Prostate Segmentation" (MICCAI 2023)**

> This repo supports source domain training, active sample selection, and target domain self-training (Stage 1 and Stage 2) on the PROMISE12 dataset.

---

## 📁 Project Structure

UGTST-reproduction/ ├── data_preprocessed/ # Preprocessed PROMISE12 data (not uploaded) ├── models/ # Trained model checkpoints ├── scripts/ # Training, selection, evaluation scripts ├── outputs/ # Logs and saved models ├── configs/ # Optional: config files ├── README.md ├── requirements.txt └── .gitignore

---

## 📦 Installation

```bash
git clone https://github.com/YOUR_NAME/UGTST-reproduction.git
cd UGTST-reproduction

conda create -n ugtst python=3.10
conda activate ugtst
pip install -r requirements.txt

---

##📂 Data Preparation
Download PROMISE12 dataset from https://promise12.grand-challenge.org/](https://liuquande.github.io/SAML/
I merged the BIDMC and HK datasets from PROMISE12 into one dataset and named it PROMISE12 in my directory.
Similarly, I merged the RUNMC and BMC datasets into one dataset and named it NCI-ISBI-2013 in my directory.
You need to modify data_dir to ./data/PROMISE12/ and ./data/NCI-ISBI-2013/ in preprocess.py to get the preprocessed data of the two datasets.
python preprocess.py
This will generate:
data_preprocessed/PROMISE12/
├── CaseXX/
│   ├── CaseXX.h5           # 3D volume with label
│   └── slices/*.h5         # 2D slices
├── slices_pseudo/          # Generated in later steps
├── target_train.txt
├── target_val.txt
├── target_test.txt
└── folds.txt

🚀 Training Pipeline
🔹 1. Train on Source Domain

python train_source.py --data_root ./data_preprocessed/NCI-ISBI-2013 \
                       --fold_file ./data_preprocessed/NCI-ISBI-2013/folds.txt \
                       --save_dir ./models/source

🔹 2. Active Sample Selection + Pseudo Label Generation

python active_select.py --model_path ./models/source/best_model.pth \
                        --target_root ./data_preprocessed/PROMISE12 \
                        --save_dir ./data_preprocessed/PROMISE12/slices_pseudo \
                        --select_ratio 0.05
This will:

Create selected.txt for most uncertain samples

Generate pseudo labels for the rest

🔹 3. Target Domain Training

python train_target.py --model_path ./models/source/best_model.pth \
                       --data_root ./data_preprocessed/PROMISE12 \
                       --pseudo_dir ./data_preprocessed/PROMISE12/slices_pseudo \
                       --active_txt ./data_preprocessed/PROMISE12/slices_pseudo/selected.txt \
                       --train_cases ./data_preprocessed/PROMISE12/target_train.txt \
                       --val_cases ./data_preprocessed/PROMISE12/target_val.txt \
                       --save_dir ./outputs/target_training \
                       --epochs_stage1 50 --epochs_stage2 50

🧪 Testing and Evaluation
python test.py --model_path ./outputs/target_training/Stage2_best_model.pth \
               --data_root ./data_preprocessed/PROMISE12 \
               --test_list ./data_preprocessed/PROMISE12/target_test.txt
You will get:

Dice coefficient and HD95 on the test set

Per-case metrics printed on screen



