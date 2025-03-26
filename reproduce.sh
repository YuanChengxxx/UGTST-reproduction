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


echo "===================="
echo "🔍 Step 3: Active Sample Selection + Pseudo Label Generation"
echo "===================="
python active_select.py \


echo "===================="
echo "📈 Step 4: Train Target Domain Model (Stage 1 + 2)"
echo "===================="
python train_target.py \


echo "===================="
echo "🧪 Step 5: Evaluate on Test Set"
echo "===================="
python test.py \
 

echo "===================="
echo "✅ All Done!"
echo "===================="
