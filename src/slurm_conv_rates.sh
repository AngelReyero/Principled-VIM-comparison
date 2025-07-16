#!/bin/bash

#SBATCH --job-name=conv_rates
#SBATCH --output=log_conv_rates%A_%a.out
#SBATCH --error=log_conv_rates%A_%a.err
#SBATCH --time=70:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=5
#SBATCH --mem=8G
#SBATCH --partition="normal,parietal"
#SBATCH --array=[101-150]%10


# Command to run
python conv_rates_par.py \
    --seeds $SLURM_ARRAY_TASK_ID
