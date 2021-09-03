#!/bin/bash
#SBATCH --job-name=VelocitySweep
#SBATCH --time=5-00:00:00
#SBATCH --error=/home/jhass2/CleanData/logs/VelocitySweep/%A-%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-10000%20
#SBATCH --output=/home/jhass2/CleanData/logs/VelocitySweep/%A-%a.err

TOPDIR=/home/jhass2/CleanData/VelocitySweep/
BETA=1.0
N_EXP=4500
NUM_OF_STEPS=100000
NUM_OF_SAVE_TIMES=2500
V_START=0.1
V_STOP=0.9
V_STEP=0.1

python3 /home/jhass2/Code/extremeDiffusion1D/runFiles/VelocityDriver.py $TOPDIR $SLURM_ARRAY_TASK_ID $BETA $N_EXP $NUM_OF_STEPS $NUM_OF_SAVE_TIMES $V_START $V_STOP $V_STEP
