#!/bin/bash
#SBATCH --job-name=DFPT28
#SBATCH --time=5-00:00:00
#SBATCH --error=/home/jhass2/jamming/JacobData/logs/FPTDiscreteTimeCorrected/%A-%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=/home/jhass2/jamming/JacobData/logs/FPTDiscreteTimeCorrected/%A-%a.out
#SBATCH --array=500-5000
#SBATCH --account=jamming
#SBATCH --partition=preempt
#SBATCH --requeue

NUM_OF_SAVE_DISTANCES=750
PROBDISTFLAG=0
BETA=1
TMAX=12000000

for N_EXP in 28
do
	TOPDIR=/home/jhass2/jamming/JacobData/FPTDiscreteTimeCorrected/$N_EXP
	mkdir -p $TOPDIR
	python3 FirstPassageTimeDiscrete.py $TOPDIR $SLURM_ARRAY_TASK_ID $BETA $N_EXP $NUM_OF_SAVE_DISTANCES $PROBDISTFLAG $TMAX
done
