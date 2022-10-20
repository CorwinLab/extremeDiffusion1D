#!/bin/bash
#SBATCH --job-name=FPTDiscrete
#SBATCH --time=7-00:00:00
#SBATCH --error=/home/jhass2/jamming/JacobData/logs/FPTDiscretePaper/%A-%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-25
#SBATCH --output=/home/jhass2/jamming/JacobData/logs/FPTDiscretePaper/%A-%a.out
#SBATCH --account=jamming
#SBATCH --partition=preempt
#SBATCH --requeue

NUM_OF_SAVE_DISTANCES=7500
PROBDISTFLAG=0
BETA=1
TMAX=3000000
PROBDISTFLAG=0

for N_EXP in 1 2 5 12 28
do
  TOPDIR=/home/jhass2/jamming/JacobData/FPTDiscretePaper/$N_EXP
  mkdir -p $TOPDIR

  python3 FirstPassageTimeDiscrete.py $TOPDIR $SLURM_ARRAY_TASK_ID $BETA $N_EXP $NUM_OF_SAVE_DISTANCES $PROBDISTFLAG $TMAX
done