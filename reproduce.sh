#!/bin/bash

echo "===================="
echo "🔧 Step 0: Activate Environment (if needed)"
echo "===================="
# conda activate ugtst  # Uncomment if using conda

echo "===================="
echo "📦 Step 1: Preprocess Data"
echo "===================="
python preprocess.py

echo "===================="
echo "🚀 Step 2: Train Source Domain Model"
echo "===================="
python train_source.py \
  --data_root ./data_preprocessed/NCI-ISBI-2013 \
  --fold_file ./data_preprocessed/NCI-ISBI-2013/folds.txt \
  --save_dir ./models/source

echo "===================="
echo "🔍 Step 3: Active Sample Selection + Pseudo Label Generation"
echo "===================="
python active_select.py \
  --model_path ./models/source/best_model.pth \
  --target_root ./data_preprocessed/PROMISE12 \
  --save_dir ./data_preprocessed/PROMISE12/slices_pseudo \
  --select_ratio 0.05

echo "===================="
echo "📈 Step 4: Train Target Domain Model (Stage 1 + 2)"
echo "===================="
python train_target.py \
  --model_path ./models/source/best_model.pth \
  --data_root ./data_preprocessed/PROMISE12 \
  --pseudo_dir ./data_preprocessed/PROMISE12/slices_pseudo \
  --active_txt ./data_preprocessed/PROMISE12/slices_pseudo/selected.txt \
  --train_cases ./data_preprocessed/PROMISE12/target_train.txt \
  --val_cases ./data_preprocessed/PROMISE12/target_val.txt \
  --save_dir ./outputs/target_training \
  --epochs_stage1 50 \
  --epochs_stage2 50

echo "===================="
echo "🧪 Step 5: Evaluate on Test Set"
echo "===================="
python test.py \
  --model_path ./outputs/target_training/Stage2_best_model.pth \
  --data_root ./data_preprocessed/PROMISE12 \
  --test_list ./data_preprocessed/PROMISE12/target_test.txt

echo "===================="
echo "✅ All Done!"
echo "===================="
