#!/bin/bash
#SBATCH --job-name=VScattering
#SBATCH --time=1-00:00:00
#SBATCH --error=/home/jhass2/jamming/JacobData/logs/ScatteringVelocities/%A-%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-500
#SBATCH --output=/home/jhass2/jamming/JacobData/logs/ScatteringVelocities/%A-%a.out
#SBATCH --account=jamming
#SBATCH --partition=preempt
#SBATCH --requeue

BETA=1

for V in 0.01 0.05 0.1 0.2 0.3 0.4 0.5 0.6
do
	TOPDIR=/home/jhass2/jamming/JacobData/ScatteringVelocities/$V
	mkdir -p $TOPDIR

	python3 scatteringVelocities.py $TOPDIR $SLURM_ARRAY_TASK_ID $BETA $V
done