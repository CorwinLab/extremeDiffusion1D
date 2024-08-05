#!/bin/bash
#SBATCH --job-name=Contin5
#SBATCH --time=0-12:00:00
#SBATCH --error=/home/jhass2/jamming/JacobData/logs/Continuous1D/%A-%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-500
#SBATCH --output=/home/jhass2/jamming/JacobData/logs/Continuous1D/%A-%a.out
#SBATCH --account=jamming
#SBATCH --partition=preempt

NEXP=4
TMAX=100000
XI=1
SIGMA=5
TOL=0.0001
D=10

TOPDIR=/home/jhass2/jamming/JacobData/Continuous1D/$D/$SIGMA/$XI
mkdir -p $TOPDIR

# (topDir, sysID, Nexp, tMax, xi, sigma, tol, D)
python3 Continuous1D.py $TOPDIR $SLURM_ARRAY_TASK_ID $NEXP $TMAX $XI $SIGMA $TOL $D
