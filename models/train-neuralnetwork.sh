#!/bin/bash

# Request an hour of runtime:
#SBATCH --time=24:00:00

# 4 cores
#SBATCH -n 4

# Ask for the GPU partition and 1 GPU
#SBATCH -p gpu --gres=gpu:1

#SBATCH --mem=40G

# Specify a job name:
#SBATCH -J NeuralNetTrain

# Specify an output file
#SBATCH -o nn_out.out
#SBATCH -e nn_out.out


# Set up the environment by loading modules
module load anaconda/3-5.2.0
module load keras/2.1.1
module load cuda/8.0.61 cudnn/5.1 tensorflow/1.5.0_gpu

source activate /users/ashah3/scratch/env

# Run a script
python -u neuralnet_fit.py > neuralnet.out
