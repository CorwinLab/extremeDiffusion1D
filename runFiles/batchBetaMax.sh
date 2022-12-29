#!/bin/bash
#SBATCH --job-name=BetaSweep
#SBATCH --time=3-00:00:00
#SBATCH --error=/home/jhass2/jamming/JacobData/logs/BetaSweepPaper/%A-%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-1000
#SBATCH --output=/home/jhass2/jamming/JacobData/logs/BetaSweepPaper/%A-%a.out
#SBATCH --account=jamming
#SBATCH --partition=preempt

NUM_OF_SAVE_TIMES=7500
PROBDISTFLAG=0
N_EXP=24

for BETA in 0.01 0.1 1 10 100
do
  TOPDIR=/home/jhass2/jamming/JacobData/BetaSweepPaper/$BETA
  mkdir -p $TOPDIR
  python3 MaxPartDriverPaper.py $TOPDIR $SLURM_ARRAY_TASK_ID $BETA $N_EXP $NUM_OF_SAVE_TIMES $PROBDISTFLAG
done
