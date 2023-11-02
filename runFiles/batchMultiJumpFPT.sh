#!/bin/bash
#SBATCH --job-name=UniformN12
#SBATCH --time=1-00:00:00
#SBATCH --error=/home/jhass2/jamming/JacobData/logs/MultiJumpRW/%A-%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-500
#SBATCH --output=/home/jhass2/jamming/JacobData/logs/MultiJumpRW/%A-%a.out
#SBATCH --account=jamming
#SBATCH --partition=preempt
#SBATCH --requeue

PREFACTOR="5e3"
SYMMETRIC="uniform"
NEXP=12
PARAMS="None"

for STEPSIZE in 3 5 11
do
	TOPDIR=/home/jhass2/jamming/JacobData/MultiJumpRWFPTPaper/$SYMMETRIC/$STEPSIZE/$NEXP/

	mkdir -p $TOPDIR

	# topDir, sysID, Lmax, step_size, symmetric, Nexp
	python3 multiJumpFPT.py $TOPDIR $SLURM_ARRAY_TASK_ID $PREFACTOR $STEPSIZE $SYMMETRIC $NEXP $PARAMS
done
