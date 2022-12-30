#!/bin/bash
#SBATCH --job-name=SCDF
#SBATCH --time=1-00:00:00
#SBATCH --error=/home/jhass2/jamming/JacobData/logs/FPTCDFSmall/%A-%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-499
#SBATCH --output=/home/jhass2/jamming/JacobData/logs/FPTCDFSmall/%A-%a.out
#SBATCH --account=jamming
#SBATCH --partition=preempt

DMAX=750
NUMOFPOINTS=750
TOPDIR=/home/jhass2/jamming/JacobData/FPTCDFSmall/
N=3
mkdir -p $TOPDIR

python3 NumbaSmallFPT.py $TOPDIR $SLURM_ARRAY_TASK_ID $DMAX $N $NUMOFPOINTS
