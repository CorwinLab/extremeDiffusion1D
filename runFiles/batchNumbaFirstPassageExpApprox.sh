#!/bin/bash
#SBATCH --job-name=ExpFCDF
#SBATCH --time=1-00:00:00
#SBATCH --error=/home/jhass2/jamming/JacobData/logs/FPTCDFPaperExp/%A-%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-1000
#SBATCH --output=/home/jhass2/jamming/JacobData/logs/FPTCDFPaperExp/%A-%a.out
#SBATCH --account=jamming
#SBATCH --partition=preempt
#SBATCH --requeue

DMAX=1000
NUMOFPOINTS=750

for N_EXP in 2
do
    TOPDIR=/home/jhass2/jamming/JacobData/FPTCDFExpApprox/$N_EXP
    mkdir -p $TOPDIR

    python3 NumbaFirstPassageExp.py $TOPDIR $SLURM_ARRAY_TASK_ID $DMAX $N_EXP $NUMOFPOINTS
done
