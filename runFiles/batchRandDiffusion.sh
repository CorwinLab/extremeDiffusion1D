#!/bin/bash
#SBATCH --job-name=RandD
#SBATCH --time=1-00:00:00
#SBATCH --error=/home/jhass2/jamming/JacobData/logs/RandomDiffusion/%A-%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-500
#SBATCH --output=/home/jhass2/jamming/JacobData/logs/RandomDiffusion/%A-%a.out
#SBATCH --account=jamming
#SBATCH --partition=preempt
#SBATCH --requeue

TOPDIR=/home/jhass2/jamming/JacobData/RandomDiffusion/
TMAX=50000
V=0.1
D0=1
SIGMA=0.1
DX=1

mkdir -p $TOPDIR 

# (topDir, sysID, tMax, v, D0, sigma, dx)
python3 RandDiffusion.py $TOPDIR $SLURM_ARRAY_TASK_ID $TMAX $V $D0 $SIGMA $DX