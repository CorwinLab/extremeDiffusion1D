#!/bin/bash
# Deleted all the SLURM stuff - you'll have to figure that out yourself
# This is modified so it doesn't go out to larger times. Shouldn't take too long
# compared to the actual script itself.

TOPDIR=./
BETA=1.0
N_EXP=4500
NUM_OF_STEPS=1000
NUM_OF_SAVE_TIMES=50
QUARTILE_START=100
QUARTILE_STOP=4500
Q_STEP=100

SLURM_ARRAY_TASK_ID=1 # This is usually made in SLURM but we'll do it here

python3 ../../runFiles/NthQuartileDriver.py $TOPDIR $SLURM_ARRAY_TASK_ID $BETA $N_EXP $NUM_OF_STEPS $NUM_OF_SAVE_TIMES $QUARTILE_START $QUARTILE_STOP $Q_STEP

# Finally make sure the files were created and then remove them
pytest testNthQuartileDriverCleanUp.py
