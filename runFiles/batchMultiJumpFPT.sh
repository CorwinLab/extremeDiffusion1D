#!/bin/bash
#SBATCH --job-name=Multi
#SBATCH --time=1-00:00:00
#SBATCH --error=/home/jhass2/jamming/JacobData/logs/MultiJumpRW/%A-%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-2
#SBATCH --output=/home/jhass2/jamming/JacobData/logs/MultiJumpRW/%A-%a.out
#SBATCH --account=jamming
#SBATCH --partition=preempt

TOPDIR=/home/jhass2/jamming/JacobData/MultiJumpRW/
LMAX=10000
STEPSIZE=5
SYMMETRIC=0
NEXP=12

# topDir, sysID, Lmax, step_size, symmetric, Nexp
python3 multiJumpFPT.py $TOPDIR $SLURM_ARRAY_TASK_ID $LMAX $STEPSIZE $SYMMETRIC $NEXP