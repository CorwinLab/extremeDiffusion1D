#!/bin/bash
#SBATCH --job-name=FPTD5
#SBATCH --time=3-00:00:00
#SBATCH --error=/home/jhass2/jamming/JacobData/logs/MultiJumpRWFPT/%A-%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-500
#SBATCH --output=/home/jhass2/jamming/JacobData/logs/MultiJumpRWFPT/%A-%a.out
#SBATCH --account=jamming
#SBATCH --partition=preempt
#SBATCH --requeue

PREFACTOR="5e4"
SYMMETRIC="dirichlet"
PARAMS="2,1,0.25,4,0.5"
STEPSIZE=5

for NEXP in 14
do
	for i in {0..4}
	do
		TOPDIR=/home/jhass2/jamming/JacobData/MultiJumpRWFPTPaper/$SYMMETRIC/$STEPSIZE/$NEXP/

		mkdir -p $TOPDIR
		id=$((SLURM_ARRAY_TASK_ID*4 + i + SLURM_ARRAY_TASK_ID))
		# topDir, sysID, Lmax, step_size, symmetric, Nexp
		python3 multiJumpFPT.py $TOPDIR $id $PREFACTOR $STEPSIZE $SYMMETRIC $NEXP $PARAMS
	done
done
