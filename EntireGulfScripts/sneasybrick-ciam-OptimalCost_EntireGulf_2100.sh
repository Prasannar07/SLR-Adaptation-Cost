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
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=10g
hostname

spack env activate default-ml-23090601
python EntireGulfScripts/sneasybrick-ciam-OptimalCost_EntireGulf_2100.py
