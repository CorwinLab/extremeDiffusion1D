#!/bin/bash
#SBATCH --job-name=2DPDF
#SBATCH --time=1-00:00:00
#SBATCH --error=/home/jhass2/jamming/JacobData/logs/2DPDF/%A-%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-500
#SBATCH --output=/home/jhass2/jamming/JacobData/logs/2DPDF/%A-%a.out
#SBATCH --account=jamming
#SBATCH --partition=preempt
#SBATCH --requeue

TMAX=3000
TOPDIR=/home/jhass2/jamming/JacobData/2DCDFOutsideR/

mkdir -p $TOPDIR

# (topDir, sysID, tMax, r)
python3 Measure2DPDF.py $TOPDIR $SLURM_ARRAY_TASK_ID $TMAX
