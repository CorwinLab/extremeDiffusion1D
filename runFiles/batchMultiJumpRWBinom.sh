#!/bin/bash
#SBATCH --job-name=betaBinom
#SBATCH --time=3-00:00:00
#SBATCH --error=/home/jhass2/jamming/JacobData/logs/MultiJumpRW/%A-%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-1000
#SBATCH --output=/home/jhass2/jamming/JacobData/logs/MultiJumpRW/%A-%a.out
#SBATCH --account=jamming
#SBATCH --partition=preempt
#SBATCH --requeue

TMAX=500000
STEPSIZE=5
DISTRIBUTION="betaBinom"
TOPDIR=/home/jhass2/jamming/JacobData/MultiJumpRWPaperSymmetric12/$DISTRIBUTION/

mkdir -p $TOPDIR

# (tMax, step_size, topDir, sysID, distribution) = sys.argv[1:]
python3 MultiJumpRWBetaBinom.py $TMAX $STEPSIZE $TOPDIR $SLURM_ARRAY_TASK_ID $DISTRIBUTION
