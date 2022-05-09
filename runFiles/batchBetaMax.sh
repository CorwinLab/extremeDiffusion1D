#!/bin/bash
#SBATCH --job-name=BetaSweep
#SBATCH --time=3-00:00:00
#SBATCH --error=/home/jhass2/jamming/JacobData/logs/BetaSweep/%A-%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-100
#SBATCH --output=/home/jhass2/jamming/JacobData/logs/BetaSweep/%A-%a.out
#SBATCH --account=jamming
#SBATCH --partition=preempt

NUM_OF_SAVE_TIMES=7500
PROBDISTFLAG=0
N_EXP=24

for BETA in 0 0.01 0.05 0.1 0.5 1 2 5 10
do
  TOPDIR=/home/jhass2/jamming/JacobData/BetaSweep/$BETA
  mkdir -p $TOPDIR
  for i in {0..10}
  do
    id=$((SLURM_ARRAY_TASK_ID*10 + i + SLURM_ARRAY_TASK_ID))
    python3 MaxPartDriverPaper.py $TOPDIR $id $BETA $N_EXP $NUM_OF_SAVE_TIMES $PROBDISTFLAG
  done
done
