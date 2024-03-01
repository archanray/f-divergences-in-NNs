#!/bin/bash

#SBATCH -c 4  # Number of Cores per Task
#SBATCH --mem=8192  # Requested Memory
#SBATCH -p gpu  # Partition
#SBATCH -G 2  # Number of GPUs
#SBATCH -t 07:00:00  # Job time limit
#SBATCH -o slurm-%j.out  # %j = job ID

python main.py
