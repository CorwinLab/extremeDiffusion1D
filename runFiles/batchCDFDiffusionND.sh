#!/bin/bash
#SBATCH --job-name=2DFPT
#SBATCH --time=3-00:00:00
#SBATCH --error=/home/jhass2/CleanData/logs/2DLattice/%A-%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-499
#SBATCH --output=/home/jhass2/CleanData/logs/2DLattice/%A-%a.out
#SBATCH --nice=1000

TOPDIR=/home/jhass2/CleanData/2DLattice/
ALPHA=inf
N=1e24
TMAX=10000
LMIN=10
LMAX=100 # This is just 100 * log(1e7)

mkdir -p $TOPDIR

python3 ./CDFFirstPassage.py $TOPDIR $SLURM_ARRAY_TASK_ID $ALPHA $N $TMAX $LMIN $LMAX