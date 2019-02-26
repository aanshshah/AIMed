#!/bin/bash

# Request an hour of runtime:
#SBATCH --time=24:00:00

# 4 cores
#SBATCH -n 2

# 20G
#SBATCH --mem=20G

# Specify a job name:
#SBATCH -J BaselineTrain

# Specify an output file
#SBATCH -o Baseline.out
#SBATCH -e Baseline.out


# Set up the environment by loading modules
module load anaconda/3-5.2.0
source activate /users/ashah3/scratch/env

# Run a script
python -u baseline_fit.py > Baseline.out
