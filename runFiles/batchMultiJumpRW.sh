#!/bin/bash
#SBATCH --job-name=VStep5
#SBATCH --time=0-06:00:00
#SBATCH --error=/home/jhass2/jamming/JacobData/logs/MultiJumpRW/%A-%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-500
#SBATCH --output=/home/jhass2/jamming/JacobData/logs/MultiJumpRW/%A-%a.out
#SBATCH --account=jamming
#SBATCH --partition=preempt

TMAX=10000
STEPSIZE=5
NEXP=28
V=0.01
DISTRIBUTION="notsymmetric"
TOPDIR=/home/jhass2/jamming/JacobData/MultiJumpRWVs/$DISTRIBUTION/$STEPSIZE/$V

# (tMax, step_size, Nexp, v, topDir, sysID, distribution) = sys.argv[1:]
python3 MultiJumpRW.py $TMAX $STEPSIZE $NEXP $V $TOPDIR $SLURM_ARRAY_TASK_ID $DISTRIBUTION