#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres gpu:1
#SBATCH -C v100
#SBATCH --job-name=pinns
#SBATCH --output=logs/output.txt

enable_lmod

module load container_env pytorch-gpu/2.8.0

crun -p ~/envs/pinns pip install -e .
srun crun -p ~/envs/pinns python3 -m pinns.burgers.continuous_inference
