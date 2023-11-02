#!/bin/bash
#SBATCH --job-name=Cyclic
#SBATCH --time=0-02:00:00
#SBATCH --error=/home/jhass2/jamming/JacobData/logs/CyclicScattering/%A-%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-250
#SBATCH --output=/home/jhass2/jamming/JacobData/logs/CyclicScattering/%A-%a.out
#SBATCH --account=jamming
#SBATCH --partition=preempt
#SBATCH --requeue

TOPDIR=/home/jhass2/jamming/JacobData/FullyConnectedScattering/
mkdir -p $TOPDIR

python3 generalizedScattering.py $TOPDIR $SLURM_ARRAY_TASK_ID
