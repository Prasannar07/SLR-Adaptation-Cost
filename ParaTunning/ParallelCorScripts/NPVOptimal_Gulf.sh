#!/bin/bash -l
#SBATCH --job-name=testJob
#SBATCH --comment="Testing Job"
#SBATCH --account=slr
#SBATCH --partition=tier3
#SBATCH --mail-user=pr1408@rit.edu
#SBATCH --mail-type=ALL
#SBATCH --time=0-30:30:00
#SBATCH --error=logs/error/%x_%j.err
#SBATCH --output=logs/output/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=10g
#SBATCH --gres=gpu:1  # Request one GPU per node

# Load CUDA module
module load cuda

# Set CUDA_VISIBLE_DEVICES if needed
# export CUDA_VISIBLE_DEVICES=0

hostname

spack env activate default-ml-23110801
python ParaTunning/ParallelCorScripts/NPVOptimal_Gulf.py