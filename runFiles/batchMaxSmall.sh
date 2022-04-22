#!/bin/bash
#SBATCH --job-name=Max2
#SBATCH --time=0-01:00:00
#SBATCH --error=/home/jhass2/jamming/JacobData/logs/Max2/%A-%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-49
#SBATCH --output=/home/jhass2/jamming/JacobData/logs/Max2/%A-%a.out
#SBATCH --account=jamming
#SBATCH --partition=preempt

NUM_OF_SAVE_TIMES=7500
PROBDISTFLAG=0
BETA=1
N_EXP=1

for i in {0..500}
do
  TOPDIR=/home/jhass2/jamming/JacobData/Max2/
  id=$((SLURM_ARRAY_TASK_ID*500 + i))
  mkdir -p $TOPDIR
  python3 MaxPartDriverPaperFixedTime.py $TOPDIR $id $BETA $N_EXP $NUM_OF_SAVE_TIMES $PROBDISTFLAG
done
