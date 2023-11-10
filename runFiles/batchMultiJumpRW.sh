#!/bin/bash
#SBATCH --job-name=Dirichlet
#SBATCH --time=1-00:00:00
#SBATCH --error=/home/jhass2/jamming/JacobData/logs/MultiJumpRW/%A-%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-499
#SBATCH --output=/home/jhass2/jamming/JacobData/logs/MultiJumpRW/%A-%a.out
#SBATCH --account=jamming
#SBATCH --partition=preempt
#SBATCH --requeue

TMAX=100000
STEPSIZE=3
NEXP=28
DISTRIBUTION="dirichlet"
TOPDIR=/home/jhass2/jamming/JacobData/MultiJumpRWPaper/$DISTRIBUTION/$STEPSIZE/$NEXP
PARAMS='12,1,12'

for i in {0..10}
do
  id=$((SLURM_ARRAY_TASK_ID*10 + i + SLURM_ARRAY_TASK_ID))

  # (tMax, step_size, Nexp, topDir, sysID, distribution, params) = sys.argv[1:]
  python3 MultiJumpRW.py $TMAX $STEPSIZE $NEXP $TOPDIR $SLURM_ARRAY_TASK_ID $DISTRIBUTION $PARAMS
done