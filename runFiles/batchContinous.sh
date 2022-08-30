#!/bin/bash
#SBATCH --job-name=Cont
#SBATCH --time=0-12:00:00
#SBATCH --error=/home/jhass2/jamming/JacobData/logs/Continuous/%A-%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-499
#SBATCH --output=/home/jhass2/jamming/JacobData/logs/Continuous/%A-%a.out
#SBATCH --account=jamming
#SBATCH --partition=preempt
#SBATCH --requeue

MINTIME=10
MAXTIME=100000
NPARTICLES=10000
NUMSAVETIMES=5000
XI=5
TOPDIR=/home/jhass2/jamming/JacobData/Continuous/

mkdir -p $TOPDIR

python3 Continuous2D.py $TOPDIR $SLURM_ARRAY_TASK_ID $MINTIME $MAXTIME $NPARTICLES $NUMSAVETIMES $XI