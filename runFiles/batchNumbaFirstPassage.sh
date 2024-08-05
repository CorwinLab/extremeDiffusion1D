#!/bin/bash
#SBATCH --job-name=FCDF
#SBATCH --time=1-00:00:00
#SBATCH --error=/home/jhass2/jamming/JacobData/logs/FPTCDF/%A-%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-500
#SBATCH --output=/home/jhass2/jamming/JacobData/logs/FPTCDF/%A-%a.out
#SBATCH --account=jamming
#SBATCH --partition=preempt
#SBATCH --requeue

DMAX=1000
NUMOFPOINTS=750
N_EXP=1

for i in {0..4}
do
    TOPDIR=/home/jhass2/jamming/JacobData/FPTCDFSam/$N_EXP
    mkdir -p $TOPDIR
    ID=$((SLURM_ARRAY_TASK_ID*4 + i + SLURM_ARRAY_TASK_ID))
    python3 NumbaFirstPassage.py $TOPDIR $ID $DMAX $N_EXP $NUMOFPOINTS
done
