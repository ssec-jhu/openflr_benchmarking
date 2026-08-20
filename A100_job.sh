#!/bin/bash
#SBATCH --job-name=a100_job
#SBATCH --partition=a100
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=2
#SBATCH --ntasks=1
#SBATCH --time=00:05:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=rhausen@jhu.edu

make all
