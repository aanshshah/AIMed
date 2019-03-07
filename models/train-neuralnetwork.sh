#!/bin/bash

# Request an hour of runtime:
#SBATCH --time=24:00:00

# 4 cores
#SBATCH -n 4

# Ask for the GPU partition and 1 GPU
#SBATCH -p gpu --gres=gpu:4

#SBATCH --mem=120G

# Specify a job name:
#SBATCH -J NeuralNetTrain

# Specify an output file
#SBATCH -o NeuralNetTrain.out
#SBATCH -e NeuralNetTrain.out

module load anaconda/3-5.2.0
source activate /gpfs/data/data2040/tf2-gpu
module unload anaconda/3-5.2.0
module load cuda/10.0.130
module load cudnn/7.4
# Run a script
python -u -W ignore train_nn_all_data.py > NeuralNetTrain.out
