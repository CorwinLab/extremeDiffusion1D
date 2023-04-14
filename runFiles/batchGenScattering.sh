#!/bin/bash
#SBATCH --job-name=GenScattering
#SBATCH --time=0-10:00:00
#SBATCH --error=/home/jhass2/jamming/JacobData/logs/GenScattering/%A-%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-500
#SBATCH --output=/home/jhass2/jamming/JacobData/logs/GenScattering/%A-%a.out
#SBATCH --account=jamming
#SBATCH --partition=preempt
#SBATCH --requeue

TOPDIR=/home/jhass2/jamming/JacobData/GeneralizedScattering/
mkdir -p $TOPDIR

python3 generalizedScattering.py $TOPDIR $SLURM_ARRAY_TASK_ID