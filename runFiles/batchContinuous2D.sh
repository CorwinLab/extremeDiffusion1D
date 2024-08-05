#!/bin/bash
#SBATCH --job-name=2DCont
#SBATCH --time=1-00:00:00
#SBATCH --error=/home/jhass2/jamming/JacobData/logs/Continuous2D/%A-%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-2
#SBATCH --output=/home/jhass2/jamming/JacobData/logs/Continuous2D/%A-%a.out
#SBATCH --account=jamming
#SBATCH --partition=preempt

NEXP=3
TMAX=1000
XI=1
SIGMA=1
TOL=0.001
D=10

TOPDIR=/home/jhass2/jamming/JacobData/Continuous2D/$D/$SIGMA/$XI

# (topDir, sysID, Nexp, tMax, xi, sigma, tol, D)
python3 Continuous2D.py $TOPDIR $SLURM_ARRAY_TASK_ID $NEXP $TMAX $XI $SIGMA $TOL $D