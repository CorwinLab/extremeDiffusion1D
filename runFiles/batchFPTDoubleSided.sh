#!/bin/bash
#SBATCH --job-name=FPT
#SBATCH --time=0-08:00:00
#SBATCH --error=/home/jhass2/jamming/JacobData/logs/FPTDoubleSided/%A-%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-499
#SBATCH --output=/home/jhass2/jamming/JacobData/logs/FPTDoubleSided/%A-%a.out
#SBATCH --account=jamming
#SBATCH --partition=preempt
#SBATCH --requeue

MINDIST=10
MAXDIST=1611
NEXP=7
NUMSAVEPOINTS=250
TOPDIR=/home/jhass2/jamming/JacobData/FPTDoubleSided/

mkdir -p $TOPDIR

python3 FirstPassageTimes.py $TOPDIR $SLURM_ARRAY_TASK_ID $MINDIST $MAXDIST $NUMSAVEPOINTS $NEXP
