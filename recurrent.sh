#!/bin/sh
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=16  # num has to be = num_env (not more than 16)
#SBATCH --partition=performance
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --array=0-3
#SBATCH --output=out/recurrent-%A_%a.out
#SBATCH --error=out/recurrent-%A_%a.err

.venv/bin/python ppo_recurrent.py --wandb_tracking --exp_name recurrent