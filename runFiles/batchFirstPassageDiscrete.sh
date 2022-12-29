#!/bin/bash
#SBATCH --job-name=DFPT
#SBATCH --time=4-00:00:00
#SBATCH --error=/home/jhass2/jamming/JacobData/logs/FPTDiscreteTimeCorrected/%A-%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=/home/jhass2/jamming/JacobData/logs/FPTDiscreteTimeCorrected/%A-%a.out
#SBATCH --array=10000-10001
#SBATCH --account=jamming
#SBATCH --partition=preempt
#SBATCH --requeue

NUM_OF_SAVE_DISTANCES=750
PROBDISTFLAG=0
BETA=1
TMAX=12000000
PROBDISTFLAG=0

for N_EXP in 5 12
do
	TOPDIR=/home/jhass2/jamming/JacobData/FPTDiscreteTimeCorrected/$N_EXP
	mkdir -p $TOPDIR
	python3 FirstPassageTimeDiscrete.py $TOPDIR $SLURM_ARRAY_TASK_ID $BETA $N_EXP $NUM_OF_SAVE_DISTANCES $PROBDISTFLAG $TMAX
done
