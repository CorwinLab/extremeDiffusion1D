#!/bin/bash
#SBATCH --job-name=LargeQuartile
#SBATCH --time=10-00:00:00
#SBATCH --error=/home/jhass2/CleanData/logs/QuartilesMillion/%A-%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-10000
#SBATCH --output=/home/jhass2/CleanData/logs/QuartilesMillion/%A-%a.err

TOPDIR=/home/jhass2/CleanData/QuartilesMillion/
BETA=1.0
N_EXP=4500
NUM_OF_STEPS=1000000
NUM_OF_SAVE_TIMES=5000
QUARTILE_START=10
QUARTILE_STOP=4500
Q_STEP=10

python3 /home/jhass2/Code/extremeDiffusion1D/runFiles/NthQuartileDriver.py $TOPDIR $SLURM_ARRAY_TASK_ID $BETA $N_EXP $NUM_OF_STEPS $NUM_OF_SAVE_TIMES $QUARTILE_START $QUARTILE_STOP $Q_STEP
