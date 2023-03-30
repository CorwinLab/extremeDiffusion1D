#!/bin/bash
#SBATCH --job-name=Scattering
#SBATCH --time=1-00:00:00
#SBATCH --error=/home/jhass2/jamming/JacobData/logs/Scattering/%A-%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-500
#SBATCH --output=/home/jhass2/jamming/JacobData/logs/Scattering/%A-%a.out
#SBATCH --account=jamming
#SBATCH --partition=preempt
#SBATCH --requeue

for BETA in 0 0.1 1 10
do
	TOPDIR=/home/jhass2/jamming/JacobData/ScatteringSweep/$BETA
	mkdir -p $TOPDIR

	python3 scatteringQuantile.py $TOPDIR $SLURM_ARRAY_TASK_ID $BETA
done