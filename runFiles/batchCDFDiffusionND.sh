#!/bin/bash
#SBATCH --job-name=2DFPT
#SBATCH --time=1-00:00:00
#SBATCH --error=/home/jhass2/jamming/JacobData/logs/2DLattice/%A-%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-1
#SBATCH --output=/home/jhass2/jamming/JacobData/logs/2DLattice/%A-%a.out
#SBATCH --nice=1000
#SBATCH --account=jamming
#SBATCH --partition=preempt

TOPDIR=/home/jhass2/jamming/JacobData/2DLatticeSSRWHigherFreq/
ALPHA=inf
N=24
TMAX=10000
LMIN=10
LMAX=7 # This is just 100 * log(1e7)

mkdir -p $TOPDIR

python3 ./CDFDiffusionND.py $TOPDIR $SLURM_ARRAY_TASK_ID $ALPHA $N $TMAX $LMIN $LMAX
