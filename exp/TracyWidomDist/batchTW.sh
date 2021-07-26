#!/bin/bash
#SBATCH --job-name=TWData
#SBATCH --time=1-00:00:00
#SBATCH --error=/home/jhass2/Data/log/TW/TWQuad.%A-%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-10000
#SBATCH --output=/home/jhass2/Data/log/TW/TWQuad.%A-%a.out

TOPDIR=/home/jhass2/Data/
python3 /home/jhass2/Code/extremeDiffusion1D/exp/TracyWidomDist/runOneExp.py $TOPDIR $SLURM_ARRAY_TASK_ID
