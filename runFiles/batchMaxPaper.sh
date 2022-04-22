#!/bin/bash
#SBATCH --job-name=EMax300
#SBATCH --time=7-00:00:00
#SBATCH --error=/home/jhass2/jamming/JacobData/logs/EinsteinPaper/%A-%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-500
#SBATCH --output=/home/jhass2/jamming/JacobData/logs/EinsteinPaper/%A-%a.out
#SBATCH --account=jamming
#SBATCH --partition=preempt

NUM_OF_SAVE_TIMES=7500
PROBDISTFLAG=0
BETA=inf

for N_EXP in 300
do
  TOPDIR=/home/jhass2/jamming/JacobData/EinsteinPaper/$N_EXP
  mkdir -p $TOPDIR
  python3 MaxPartDriverPaper.py $TOPDIR $SLURM_ARRAY_TASK_ID $BETA $N_EXP $NUM_OF_SAVE_TIMES $PROBDISTFLAG
done
