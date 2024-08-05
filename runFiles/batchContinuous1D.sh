#!/bin/bash
#SBATCH --job-name=Contin
#SBATCH --time=1-00:00:00
#SBATCH --error=/home/jhass2/jamming/JacobData/logs/Continuous1D/%A-%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-2
#SBATCH --output=/home/jhass2/jamming/JacobData/logs/Continuous1D/%A-%a.out
#SBATCH --account=jamming
#SBATCH --partition=preempt

NEXP=5
TMAX=100000
XI=1
SIGMA=1
TOL=0.0001
D=10

TOPDIR=/home/jhass2/jamming/JacobData/Continuous1D/$D/$SIGMA/$XI

# (topDir, sysID, Nexp, tMax, xi, sigma, tol, D)
python3 multiJumpFPT.py $TOPDIR $SLURM_ARRAY_TASK_ID $NEXP $TMAX $XI $SIGMA $TOL $D