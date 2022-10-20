#!/bin/bash
#SBATCH --job-name=FCDF
#SBATCH --time=1-00:00:00
#SBATCH --error=/home/jhass2/jamming/JacobData/logs/FixedFirstPassNDependence/%A-%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-499
#SBATCH --output=/home/jhass2/jamming/JacobData/logs/FixedFirstPassNDependence/%A-%a.out
#SBATCH --account=jamming
#SBATCH --partition=preempt

DISTANCE=500
NMIN=2
NMAX=300
NUMOFPOINTS=100
TOPDIR=/home/jhass2/jamming/JacobData/FixedFirstPassNDependence/

mkdir -p $TOPDIR

python3 NumbaFirstNDepend.py $TOPDIR $SLURM_ARRAY_TASK_ID $DISTANCE $NMIN $NMAX $NUMOFPOINTS