#!/bin/bash
#SBATCH --job-name=FPT
#SBATCH --time=0-01:00:00
#SBATCH --error=/home/jhass2/jamming/JacobData/logs/FirstPassTest/%A-%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-499
#SBATCH --output=/home/jhass2/jamming/JacobData/logs/FirstPassTest/%A-%a.out
#SBATCH --account=jamming
#SBATCH --partition=preempt

TOPDIR=/home/jhass2/jamming/JacobData/FirstPassTest/
DISTANCE=1611 # This is just 100 * log(1e7)
NEXP=7

for i in {0..10}
do
    id=$((SLURM_ARRAY_TASK_ID*10 + i + SLURM_ARRAY_TASK_ID))
    python3 ./CDFFirstPassage.py $TOPDIR $id $DISTANCE $NEXP
done