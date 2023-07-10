#!/bin/bash
#SBATCH --job-name=Cont1D
#SBATCH --time=7-00:00:00
#SBATCH --error=/home/jhass2/CleanData/logs/ContPeter/%A-%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-499
#SBATCH --output=/home/jhass2/CleanData/logs/ContPeter/%A-%a.out
#SBATCH --account=jamming
#SBATCH --partition=preempt

MINTIME=1
MAXTIME=100
NPARTICLES=1000000
NUMSAVETIMES=2500
XI=1

for D in 1
do
	TOPDIR=/home/jhass2/CleanData/ContPeter/$D

	mkdir -p $TOPDIR

	python3 Continuous1D.py $TOPDIR 0 $MINTIME $MAXTIME $NPARTICLES $NUMSAVETIMES $XI $D
done
