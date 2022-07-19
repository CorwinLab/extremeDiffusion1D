#!/bin/bash
#SBATCH --job-name=FirstPass
#SBATCH --time=0-12:00:00
#SBATCH --error=/home/jhass2/jamming/JacobData/logs/FirstPassCDF/%A-%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-49
#SBATCH --output=/home/jhass2/jamming/JacobData/logs/FirstPassCDF/%A-%a.out
#SBATCH --account=jamming
#SBATCH --partition=preempt

TOPDIR=/home/jhass2/jamming/JacobData/FirstPassCDF/
BETA=1
NEXP=24
DMIN=50
DMAX=500
CUTOFF=0.99

mkdir -p $TOPDIR

for i in {0..10}
do
    id=$((SLURM_ARRAY_TASK_ID*10 + i + SLURM_ARRAY_TASK_ID))
    python3 FirstPassageTimes.py $TOPDIR $BETA $NEXP $id $DMIN $DMAX $CUTOFF
done
