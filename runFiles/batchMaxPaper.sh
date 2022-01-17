#!/bin/bash
#SBATCH --job-name=PaperMax
#SBATCH --time=30-00:00:00
#SBATCH --error=/home/jhass2/CleanData/logs/Paper/Max/%A-%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-250
#SBATCH --output=/home/jhass2/CleanData/logs/Paper/Max/%A-%a.out
#SBATCH --nice=1000

NUM_OF_SAVE_TIMES=7500
PROBDISTFLAG=0
BETA=1

for N_EXP in 2 7 24 85 300
do
  TOPDIR=/home/jhass2/CleanData/Paper/Max/$N_EXP
  mkdir -p $TOPDIR
  python3 MaxPartDriverPaper.py $TOPDIR $SLURM_ARRAY_TASK_ID $BETA $N_EXP $NUM_OF_SAVE_TIMES $PROBDISTFLAG
done
