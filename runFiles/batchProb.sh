#!/bin/bash
#SBATCH --job-name=ProbVel
#SBATCH --time=0-00:30:00
#SBATCH --error=/home/jhass2/jamming/JacobData/logs/ProbVel/%A-%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-99
#SBATCH --output=/home/jhass2/jamming/JacobData/logs/ProbVel/%A-%a.out
#SBATCH	--account=jamming
#SBATCH --partition=preempt

TOP_DIR=/home/jhass2/jamming/JacobData/ProbVel
BETA=1
QUANTILE=50
mkdir -p $TOP_DIR

for i in {0..100}
do
	id=$((SLURM_ARRAY_TASK_ID*100 + i)) 
	python3 ProbAndV.py $TOP_DIR $id $BETA $QUANTILE
done
