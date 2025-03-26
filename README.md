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
