#!/bin/bash
#SBATCH --job-name=Cont
#SBATCH --time=1-00:00:00
#SBATCH --error=/home/jhass2/CleanData/logs/ContinuousCorrected/%A-%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-499
#SBATCH --output=/home/jhass2/CleanData/logs/ContinuousCorrected/%A-%a.out

MINTIME=1
MAXTIME=10000
NPARTICLES=100000
NUMSAVETIMES=2500
XI=5
D=1
TOPDIR=/home/jhass2/CleanData/ContinuousCorrected/

mkdir -p $TOPDIR

python3 Continuous1D.py $TOPDIR $SLURM_ARRAY_TASK_ID $MINTIME $MAXTIME $NPARTICLES $NUMSAVETIMES $XI $D