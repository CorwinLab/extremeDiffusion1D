#!/bin/bash
#SBATCH --job-name=Multi
#SBATCH --time=1-00:00:00
#SBATCH --error=/home/jhass2/jamming/JacobData/logs/MultiJumpRW/%A-%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-500
#SBATCH --output=/home/jhass2/jamming/JacobData/logs/MultiJumpRW/%A-%a.out
#SBATCH --account=jamming
#SBATCH --partition=preempt
#SBATCH --requeue

TMAX=100000
STEPSIZE=11
NEXP=12
V=0.5
TOPDIR=/home/jhass2/jamming/JacobData/MultiJumpRW/

#(tMax, step_size, Nexp, v, topDir, sysID) = sys.argv[1:]
python3 MultiJumpRW.py $TMAX $STEPSIZE $NEXP $V $TOPDIR $SLURM_ARRAY_TASK_ID