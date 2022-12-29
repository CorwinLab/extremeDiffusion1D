#!/bin/bash
#SBATCH --job-name=ContD
#SBATCH --time=3-00:00:00
#SBATCH --error=/home/jhass2/CleanData/logs/Continuous/%A-%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-499
#SBATCH --output=/home/jhass2/CleanData/logs/Continuous/%A-%a.out

MINTIME=1
MAXTIME=10000
NPARTICLES=100000
NUMSAVETIMES=2500
XI=1

for D in 5 10 20
do
	TOPDIR=/home/jhass2/CleanData/Continuous/$XI/$D

	mkdir -p $TOPDIR

	python3 Continuous1D.py $TOPDIR $SLURM_ARRAY_TASK_ID $MINTIME $MAXTIME $NPARTICLES $NUMSAVETIMES $XI $D
done
