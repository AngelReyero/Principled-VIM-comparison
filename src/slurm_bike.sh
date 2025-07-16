#!/bin/bash

#SBATCH --job-name=bike
#SBATCH --output=log_bike%A_%a.out
#SBATCH --error=log_bike%A_%a.err
#SBATCH --time=40:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=5
#SBATCH --mem=8G
#SBATCH --partition="normal,parietal"
#SBATCH --array=[1-50]%5


# Command to run
python bike_sharing_par.py \
    --seeds $SLURM_ARRAY_TASK_ID
