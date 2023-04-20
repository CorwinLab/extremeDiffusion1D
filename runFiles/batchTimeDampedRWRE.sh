#!/bin/bash
#SBATCH --job-name=VScattering
#SBATCH --time=0-05:00:00
#SBATCH --error=/home/jhass2/jamming/JacobData/logs/RWREDamped/%A-%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-500
#SBATCH --output=/home/jhass2/jamming/JacobData/logs/RWREDamped/%A-%a.out
#SBATCH --account=jamming
#SBATCH --partition=preempt
#SBATCH --requeue

GAMMA=0.5
TMAX=1e5
TOPDIR=/home/jhass2/jamming/JacobData/RWREDamped/$GAMMA
mkdir -p $TOPDIR

python3 scatteringVelocities.py $TOPDIR $SLURM_ARRAY_TASK_ID $TMAX $GAMMA
