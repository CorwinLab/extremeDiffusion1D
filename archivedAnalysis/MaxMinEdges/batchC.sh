#!/bin/bash
#SBATCH --job-name=generateDiffusionData
#SBATCH --time=0-12:00:00
#SBATCH --error=/home/jhass2/Data/log/generateDiffusionData.%A-%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-100
#SBATCH --output=/home/jhass2/Data/log/generateDiffusionData.%A-%a.out

TOPDIR=/home/jhass2/Data/
NUM=1e300

for beta in 0.25 0.5 1 1.5 2 5 10
do 
	python3 /home/jhass2/Code/extremeDiffusion1D/exp/getCdata.py $TOPDIR $NUM $beta $SLURM_ARRAY_TASK_ID
done
