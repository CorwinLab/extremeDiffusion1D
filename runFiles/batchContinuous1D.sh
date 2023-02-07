#!/bin/bash
#SBATCH --job-name=Cont1D
#SBATCH --time=7-00:00:00
#SBATCH --error=/home/jhass2/jamming/JacobData/logs/ContinuousNew/%A-%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-499
#SBATCH --output=/home/jhass2/jamming/JacobData/logs/ContinuousNew/%A-%a.out
#SBATCH --account=jamming
#SBATCH --partition=preempt

MINTIME=1
MAXTIME=1000000
NPARTICLES=1000000
NUMSAVETIMES=2500
XI=1
SIGMA=1

for D in 0.1 1 10
do
	TOPDIR=/home/jacob/Desktop/Code/extremeDiffusion1D/testContinuous/

	mkdir -p $TOPDIR

	python3 Continuous1D.py $TOPDIR 0 $MINTIME $MAXTIME $NPARTICLES $NUMSAVETIMES $XI $D $SIGMA
done
