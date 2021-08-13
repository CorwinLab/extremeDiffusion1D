#!/bin/bash
#SBATCH --job-name=LargeQuartile
#SBATCH --time=8-00:00:00
#SBATCH --error=/home/jhass2/Data/log/N4500.%A-%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-2500
#SBATCH --output=/home/jhass2/Data/log/N4500.%A-%a.out

TOPDIR=/home/jhass2/Data/
python3 /home/jhass2/Code/extremeDiffusion1D/exp/NthQuartileTurnOn/N8000DataScript.py $TOPDIR $SLURM_ARRAY_TASK_ID
