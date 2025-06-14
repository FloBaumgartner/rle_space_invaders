#!/bin/sh
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=16  # num has to be = num_env (not more than 16)
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --output=out/impala-%A_%a.out
#SBATCH --error=out/impala-%A_%a.err

.venv/bin/python ppo_impala.py --wandb_tracking --exp_name impala