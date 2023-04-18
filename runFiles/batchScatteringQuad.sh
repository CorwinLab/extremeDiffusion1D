#!/bin/bash
#SBATCH --job-name=VScattering
#SBATCH --time=0-10:00:00
#SBATCH --error=/home/jhass2/jamming/JacobData/logs/ScatteringVelocities/%A-%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-5000
#SBATCH --output=/home/jhass2/jamming/JacobData/logs/ScatteringVelocities/%A-%a.out
#SBATCH --account=jamming
#SBATCH --partition=preempt

TMAX=1e5
TOPDIR=/home/jhass2/jamming/JacobData/ScatteringVelocitiesQuads/
mkdir -p $TOPDIR

python3 scatteringVelocities.py $TOPDIR $SLURM_ARRAY_TASK_ID $TMAX