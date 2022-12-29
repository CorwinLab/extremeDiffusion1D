#!/bin/bash
#SBATCH --job-name=BSweep
#SBATCH --time=1-00:00:00
#SBATCH --error=/home/jamming/JacobData/logs/CDFBetaSweep/%A-%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-1000
#SBATCH --output=/home/jamming/JacobData/logs/CDFBetaSweep/%A-%a.out
#SBATCH --account=jamming
#SBATCH --queue=preempt

NUM_OF_SAVE_TIMES=7500

for BETA in 0.01 0.1 1 10 100
do 
	TOPDIR=/home/jhass2/jamming/JacobData/CDFBetaSweep/$BETA
	mkdir -p $TOPDIR
	python3 ./BetaCDF.py $TOPDIR $SLURM_ARRAY_TASK_ID $BETA $NUM_OF_SAVE_TIMES
done
