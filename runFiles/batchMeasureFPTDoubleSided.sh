#!/bin/bash
#SBATCH --job-name=FPTCDF
#SBATCH --time=7-00:00:00
#SBATCH --error=/home/jhass2/jamming/JacobData/logs/FPTCDF/%A-%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-50
#SBATCH --output=/home/jhass2/jamming/JacobData/logs/FPTCDF/%A-%a.out
#SBATCH --account=jamming
#SBATCH --partition=preempt
#SBATCH --requeue

TOPDIR=/home/jhass2/jamming/JacobData/FPTCDF/
DISTANCE=1000
TMAX=3900000
NUM_OF_POINTS=7500

mkdir -p $TOPDIR

python3 ./measureFPTDoubleSided.py $TOPDIR $DISTANCE $TMAX $NUM_OF_POINTS $SLURM_ARRAY_TASK_ID
