#!/bin/bash
#SBATCH --job-name=FirstPass
#SBATCH --time=0-12:00:00
#SBATCH --error=/home/jhass2/jamming/JacobData/logs/ParallelFirstPass/%A-%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-10
#SBATCH --output=/home/jhass2/jamming/JacobData/logs/ParallelFirstPass/%A-%a.out
#SBATCH --account=jamming
#SBATCH --partition=preempt

NUM_OF_SAVE_DISTANCES=1500
BETA=inf
TMAX=500000
NUM_OF_SYSTEMS=50

for N_EXP in 2 7 24 85
do
  TOPDIR=/home/jhass2/jamming/JacobData/FirstPassAbsTimed/$N_EXP/
  mkdir -p $TOPDIR
  for i in {0..5}
  do
    id=$((SLURM_ARRAY_TASK_ID*5 + i + SLURM_ARRAY_TASK_ID))
    python3 FirstPassageTime.py $TOPDIR $id $BETA $N_EXP $NUM_OF_SAVE_DISTANCES $NUM_OF_SYSTEMS $TMAX
  done
done
