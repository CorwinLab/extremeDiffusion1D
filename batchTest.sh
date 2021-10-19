#!/bin/bash
#SBATCH --job-name=Test
#SBATCH --time=00-01:00:00
#SBATCH --error=/home/jhass2/CleanData/logs/%A-%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --array=0
#SBATCH --output=/home/jhass2/CleanData/logs/%A-%a.err

python3 ./locustSaveTime.py
