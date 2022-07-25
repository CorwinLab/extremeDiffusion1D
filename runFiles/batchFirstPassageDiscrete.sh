#!/bin/bash
#SBATCH --job-name=QFirstPass
#SBATCH --time=1-00:00:00
#SBATCH --error=/home/jhass2/jamming/JacobData/logs/QuenchedFirstPassageDiscrete/%A-%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-25
#SBATCH --output=/home/jhass2/jamming/JacobData/logs/QuenchedFirstPassageDiscrete/%A-%a.out
#SBATCH --account=jamming
#SBATCH --partition=preempt

NUM_OF_SAVE_DISTANCES=7500
PROBDISTFLAG=0
BETA=inf
TMAX=3000000
PROBDISTFLAG=0
STATICENVIRONMENT=1
N_EXP=24
TOPDIR=/home/jhass2/jamming/JacobData/QuenchedFirstPassageDiscrete/
mkdir -p $TOPDIR
for i in {0..20}
do
  id=$((SLURM_ARRAY_TASK_ID*20 + i + SLURM_ARRAY_TASK_ID))
  python3 FirstPassageTimeDiscrete.py $TOPDIR $id $BETA $N_EXP $NUM_OF_SAVE_DISTANCES $PROBDISTFLAG $TMAX $STATICENVIRONMENT
done
