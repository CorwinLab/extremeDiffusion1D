#!/bin/bash
#SBATCH --job-name=Symm5
#SBATCH --time=0-12:00:00
#SBATCH --error=/home/jhass2/jamming/JacobData/logs/MultiJumpRW/%A-%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-500
#SBATCH --output=/home/jhass2/jamming/JacobData/logs/MultiJumpRW/%A-%a.out
#SBATCH --account=jamming
#SBATCH --partition=preempt
#SBATCH --requeue

TMAX=10000
DISTRIBUTION="thirdMoment"
PARAMS='None'
STEPSIZE=5

for NEXP in 28
do
  for i in {0..4}
  do
    id=$((SLURM_ARRAY_TASK_ID*4 + i + SLURM_ARRAY_TASK_ID))
  
    TOPDIR=/home/jhass2/jamming/JacobData/MultiJumpRWPaper/$DISTRIBUTION/$STEPSIZE/$NEXP/
    mkdir -p $TOPDIR

    # (tMax, step_size, Nexp, topDir, sysID, distribution, params) = sys.argv[1:]
    python3 MultiJumpRW.py $TMAX $STEPSIZE $NEXP $TOPDIR $id $DISTRIBUTION $PARAMS
done
done
