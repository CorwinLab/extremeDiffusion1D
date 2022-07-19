#!/bin/bash
#SBATCH --job-name=NFirstPass
#SBATCH --time=0-12:00:00
#SBATCH --error=/home/jhass2/jamming/JacobData/logs/NFirstPassCDF/%A-%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-49
#SBATCH --output=/home/jhass2/jamming/JacobData/logs/NFirstPassCDF/%A-%a.out
#SBATCH --account=jamming
#SBATCH --partition=preempt

TOPDIR=/home/jhass2/jamming/JacobData/NFirstPassCDF/
BETA=1
DISTANCE=700
N_MIN=10
N_MAX=300
NUMBER_OF_NS=15

mkdir -p $TOPDIR

for i in {0..10}
do
    id=$((SLURM_ARRAY_TASK_ID*10 + i + SLURM_ARRAY_TASK_ID))
    python3 FirstPassageNDependence.py $TOPDIR $id $BETA $DISTANCE $CUTOFF $N_MIN $N_MAX $NUMBER_OF_NS
done
