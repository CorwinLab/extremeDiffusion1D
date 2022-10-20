#!/bin/bash
#SBATCH --job-name=FPT
#SBATCH --time=1-00:00:00
#SBATCH --error=/home/jhass2/CleanData/logs/FirstPassTest/%A-%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-499
#SBATCH --output=/home/jhass2/CleanData/logs/FirstPassTest/%A-%a.out
#SBATCH --nice=1000

TOPDIR=/home/jhass2/CleanData/FirstPassTest
DMIN=10
DMAX=1611 # This is just 100 * log(1e7)
NUMOFPOINTS=250
NEXP=7

mkdir -p $TOPDIR

python3 ./CDFFirstPassage.py $TOPDIR $SLURM_ARRAY_TASK_ID $DMIN $DMAX $NUMOFPOINTS $NEXP