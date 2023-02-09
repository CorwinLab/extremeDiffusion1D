#!/bin/bash
#SBATCH --job-name=2DFPT
#SBATCH --time=1-00:00:00
#SBATCH --error=/home/jhass2/jamming/JacobData/logs/2DLatticeSpherical/%A-%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-1000
#SBATCH --output=/home/jhass2/jamming/JacobData/logs/2DLatticeSpherical/%A-%a.out
#SBATCH --account=jamming
#SBATCH --queue=preempt

TOPDIR=/home/jhass2/jamming/JacobData/2DLatticeRWRESpherical/
ALPHA=1
N=1e24
RMIN=10
RMAX=1000

mkdir -p $TOPDIR

python3 ./CDFFirstPassage.py $TOPDIR $SLURM_ARRAY_TASK_ID $ALPHA $N $RMIN $RMAX