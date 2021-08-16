#!/bin/bash

TOPDIR=./
BETA=1.0
N_EXP=4500
NUM_OF_STEPS=1000
NUM_OF_SAVE_TIMES=100
V_START=0.1
V_STOP=0.9
V_STEP=0.1

SLURM_ARRAY_TASK_ID=1 # Usually make this in SLRUM but for this test we'll just create it ourselves

python3 ../runFiles/VelocityDriver.py $TOPDIR $SLURM_ARRAY_TASK_ID $BETA $N_EXP $NUM_OF_STEPS $NUM_OF_SAVE_TIMES $V_START $V_STOP $V_STEP

# Finally make sure the files were created and then remove them
pytest testVelocityDriverCleanUp.py
