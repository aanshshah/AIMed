#!/bin/bash

# Request an hour of runtime:
#SBATCH --time=24:00:00

# 4 cores
#SBATCH -n 4

# Ask for the GPU partition and 1 GPU

#SBATCH --mem=40G

# Specify a job name:
#SBATCH -J NeuralNetTrain

# Specify an output file
#SBATCH -o nn_out.out
#SBATCH -e nn_out.out


module load anaconda/3-5.2.0
source activate /gpfs/data/data2040/tf2-gpu
module unload anaconda/3-5.2.0
module load cuda/10.0.130
module load cudnn/7.4
# Run a script
python -u neuralnet_fit.py > neuralnet.out
