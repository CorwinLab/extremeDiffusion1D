#!/bin/bash
#SBATCH --job-name=Dirichlet
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
NEXP=12
DISTRIBUTION="dirichlet"
TOPDIR=/home/jhass2/jamming/JacobData/MultiJumpRW/$DISTRIBUTION/$STEPSIZE/$NEXP
PARAMS='1,5,1,5,1'

# (tMax, step_size, Nexp, topDir, sysID, distribution, params) = sys.argv[1:]
python3 MultiJumpRW.py $TMAX $STEPSIZE $NEXP $TOPDIR $SLURM_ARRAY_TASK_ID $DISTRIBUTION $PARAMS