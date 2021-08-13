#!/bin/bash
#SBATCH --job-name=LargeQuartile
#SBATCH --time=8-00:00:00
#SBATCH --error=/home/jhass2/Data/log/test/test.%A-%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-100
#SBATCH --output=/home/jhass2/Data/log/test/test.%A-%a.err

TOPDIR=/home/jhass2/Data/
BETA=1.0
N_EXP=4500
NUM_OF_STEPS=10000
NUM_OF_SAVE_TIMES=5000
QUARTILE_START=50
QUARTILE_STOP=4500
Q_STEP=50

python3 /home/jhass2/Code/extremeDiffusion1D/exp/NthQuartileTurnOn/N8000DataScript.py $TOPDIR $SLURM_ARRAY_TASK_ID $BETA $N_EXP $NUM_OF_STEPS $NUM_OF_SAVE_TIMES $QUARTILE_START $QUARTILE_STOP $Q_STEP
