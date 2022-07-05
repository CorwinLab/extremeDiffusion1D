#!/bin/bash
#SBATCH --job-name=DiffCDF
#SBATCH --time=3-00:00:00
#SBATCH --error=/home/jhass2/jamming/JacobData/logs/BetaCDFLongTime/%A-%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-50
#SBATCH --output=/home/jhass2/jamming/JacobData/logs/BetaCDFLongTime/%A-%a.out
#SBATCH --account=jamming
#SBATCH --partition=preempt

NUM_OF_SAVE_TIMES=7500

for BETA in 0.01 0.1 1 10 100
do 
	TOPDIR=/home/jhass2/jamming/JacobData/BetaCDFLongTime/$BETA
	mkdir -p $TOPDIR
	for i in {0..10}
	do
		id=$((SLURM_ARRAY_TASK_ID*10 + i + SLURM_ARRAY_TASK_ID))
		python3 ./BetaCDF.py $TOPDIR $id $BETA $NUM_OF_SAVE_TIMES
	done
done
