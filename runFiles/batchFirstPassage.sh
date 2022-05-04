#!/bin/bash
#SBATCH --job-name=FirstPass
#SBATCH --time=1-00:00:00
#SBATCH --error=/home/jhass2/jamming/JacobData/logs/FirstPass/%A-%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-100
#SBATCH --output=/home/jhass2/jamming/JacobData/logs/FirstPass/%A-%a.out
#SBATCH --account=jamming
#SBATCH --partition=preempt

NUM_OF_SAVE_DISTANCES=7500
PROBDISTFLAG=0
BETA=inf
N_EXP=24
TMAX=1000000
MAX_DISTANCE=2000
PROBDISTFLAG=0
TOPDIR=/home/jhass2/jamming/JacobData/FirstPass/

mkdir -p $TOPDIR

for i in {0..10}
do
  id=$((SLURM_ARRAY_TASK_ID*10 + i + SLURM_ARRAY_TASK_ID))
  python3 MaxPartDriverPaper.py $TOPDIR $SLURM_ARRAY_TASK_ID $BETA $N_EXP $NUM_OF_SAVE_DISTANCES $MAX_DISTANCE $PROBDISTFLAG $TMAX
done
