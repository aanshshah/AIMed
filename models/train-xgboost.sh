#!/bin/bash

# Request an hour of runtime:
#SBATCH --time=00:60:00

# Ask for the GPU partition and 1 GPU
#SBATCH -p gpu --gres=gpu:1

# 4 cores
#SBATCH -n 4

# 20G
#SBATCH --mem=20G

# Specify a job name:
#SBATCH -J XGBoostTrain

# Specify an output file
#SBATCH -o XGBoost.out
#SBATCH -e XGBoost.out


# Set up the environment by loading modules
module load anaconda/3-5.2.0
module load cuda/8.0.61 cudnn/5.1
source activate /users/ashah3/scratch/env

# Run a script
python -u xg_boost_fit.py > XGBoost.out
