#!/bin/bash
#SBATCH --job-name=UFirstPass
#SBATCH --time=0-12:00:00
#SBATCH --error=/home/jhass2/jamming/JacobData/logs/FirstPassAbsTimed/%A-%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-100
#SBATCH --output=/home/jhass2/jamming/JacobData/logs/FirstPassAbsTimed/%A-%a.out
#SBATCH --account=jamming
#SBATCH --partition=preempt

NUM_OF_SAVE_DISTANCES=7500
PROBDISTFLAG=0
BETA=1
TMAX=3000000
PROBDISTFLAG=0

for N_EXP in 2 7 24 85
do
  TOPDIR=/home/jhass2/jamming/JacobData/FirstPassAbsTimed/$N_EXP/
  mkdir -p $TOPDIR
  for i in {0..25}
  do
    id=$((SLURM_ARRAY_TASK_ID*25 + i + SLURM_ARRAY_TASK_ID))
    python3 FirstPassageTime.py $TOPDIR $id $BETA $N_EXP $NUM_OF_SAVE_DISTANCES $PROBDISTFLAG $TMAX
  done
done
