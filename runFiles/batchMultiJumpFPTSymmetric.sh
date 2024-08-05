#!/bin/bash
#SBATCH --job-name=ConstDiff
#SBATCH --time=3-00:00:00
#SBATCH --error=/home/jhass2/jamming/JacobData/logs/MultiJumpRWFPTSymmetric/%A-%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-500
#SBATCH --output=/home/jhass2/jamming/JacobData/logs/MultiJumpRWFPTSymmetric/%A-%a.out
#SBATCH --account=jamming
#SBATCH --partition=preempt
#SBATCH --requeue

PREFACTOR="5e4"
SYMMETRIC="constDiffusionCoefficient"
NEXP=28
PARAMS="None"
STEPSIZE=21
TOPDIR=/home/jhass2/jamming/JacobData/MultiJumpRWFPTPaper/$SYMMETRIC/$STEPSIZE/$NEXP/
mkdir -p $TOPDIR

for i in {0..4}
do 
	id=$((SLURM_ARRAY_TASK_ID*4 + i + SLURM_ARRAY_TASK_ID))
	# topDir, sysID, Lmax, step_size, symmetric, Nexp
	
	python3 multiJumpFPTSymmetric.py $TOPDIR $id $PREFACTOR $STEPSIZE $SYMMETRIC $NEXP $PARAMS
done
