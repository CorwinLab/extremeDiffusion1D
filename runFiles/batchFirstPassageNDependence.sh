#!/bin/bash
#SBATCH --job-name=NFirstPass
#SBATCH --time=0-06:00:00
#SBATCH --error=/home/jhass2/jamming/JacobData/logs/MultipleNFirstPassCDF/%A-%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=/home/jhass2/jamming/JacobData/logs/MultipleNFirstPassCDF/%A-%a.out
#SBATCH --account=jamming
#SBATCH --partition=preempt
#SBATCH --nice=10000

BETA=1
DISTANCE=500
N_MIN=10
N_MAX=300
NUMBER_OF_NS=75
CUTOFF=1
TOPDIR=/home/jhass2/jamming/JacobData/MultipleNFirstPassCDF/$DISTANCE
mkdir -p $TOPDIR

for i in {0..500}
do
    id=$((SLURM_ARRAY_TASK_ID + i + SLURM_ARRAY_TASK_ID))
    python3 FirstPassageNDependence.py $TOPDIR $id $BETA $DISTANCE $CUTOFF $N_MIN $N_MAX $NUMBER_OF_NS
done
