#!/bin/bash

#SBATCH --job-name=bike
#SBATCH --output=logfolder/log_bike_%A_%a.out
#SBATCH --error=logfolder/log_bike_%A_%a.err
#SBATCH --time=50:00:00
#SBATCH --cpus-per-task=5
#SBATCH --mem=8G
#SBATCH --partition=normal,parietal
#SBATCH --array=0-59%5

# ---- Arrays ----
DATASETS=(0 0.3 0.6 0.9)

MODELS=(
    lr lasso dt rf et gb hgb ab bag mlp svr knn xgb SuperLearner TabICL
) # 15

NUM_SEEDS=1

seed=$((10+SLURM_ARRAY_TASK_ID % NUM_SEEDS))

dataset_idx=$((SLURM_ARRAY_TASK_ID % ${#DATASETS[@]}))
model_idx=$((SLURM_ARRAY_TASK_ID % ${#MODELS[@]}))

corr=${DATASETS[$dataset_idx]}
mod=${MODELS[$model_idx]}

echo "Seed: $seed"
echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "Correlation: $corr"
echo "Model: $mod"

# ---- Run ----
python bike_sharing_par.py \
    --seeds $seed \
    --correlation $corr \
    --model $mod

echo "Finished model=$mod correlation=$corr seed=$seed"