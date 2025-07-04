#!/bin/bash

# Request an hour of runtime:
#SBATCH --time=24:00:00

# 4 cores
#SBATCH -n 6

#SBATCH --mem=120G

# Specify a job name:
#SBATCH -J XGBoostAll

# Specify an output file
#SBATCH -o XGBoostAll.out
#SBATCH -e XGBoostAll.out

module load anaconda/3-5.2.0
source activate /gpfs/data/data2040/tf2
# Run a script
python -u -W ignore xg_boost_fit_all_data.py > XGBoostAll.out
