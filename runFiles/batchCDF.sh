#!/bin/bash
#SBATCH --job-name=SmallCDF
#SBATCH --time=0-00:45:00
#SBATCH --error=/home/jhass2/jamming/JacobData/logs/SmallCDF/%A-%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-100
#SBATCH --output=/home/jhass2/jamming/JacobData/logs/SmallCDF/%A-%a.err
#SBATCH --account=jamming
#SBATCH --partition=preempt

TOPDIR=/home/jhass2/jamming/JacobData/SmallCDF/
BETA=1.0
TMAX=100000
NUM_OF_SAVE_TIMES=7500
QUARTILE_START=2
QUARTILE_STOP=21
QUARTILE_STEP=1

for i in {0..5}
do
	mkdir -p $TOPDIR
	id=$((SLURM_ARRAY_TASK_ID*5 + i))
	python3 ./CDFDriver.py $TOPDIR $id $BETA $TMAX $NUM_OF_SAVE_TIMES $QUARTILE_START $QUARTILE_STOP $QUARTILE_STEP
done
