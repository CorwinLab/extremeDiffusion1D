#!/bin/bash
#SBATCH --job-name=Scattering
#SBATCH --time=0-12:00:00
#SBATCH --error=/home/jhass2/jamming/JacobData/logs/ScatteringModified/%A-%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-500
#SBATCH --output=/home/jhass2/jamming/JacobData/logs/ScatteringModified/%A-%a.out
#SBATCH --account=jamming
#SBATCH --partition=preempt

BETA=1

for NEXP in 5
do
	TOPDIR=/home/jhass2/jamming/JacobData/ScatteringModified/$NEXP
	mkdir -p $TOPDIR

	python3 scatteringQuantile.py $TOPDIR $SLURM_ARRAY_TASK_ID $BETA $NEXP
done
