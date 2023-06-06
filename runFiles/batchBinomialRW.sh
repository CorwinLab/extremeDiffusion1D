#!/bin/bash
#SBATCH --job-name=Binom
#SBATCH --time=1-00:00:00
#SBATCH --error=/home/jhass2/jamming/JacobData/logs/Binom/%A-%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-500
#SBATCH --output=/home/jhass2/jamming/JacobData/logs/Binom/%A-%a.out
#SBATCH --account=jamming
#SBATCH --partition=preempt
#SBATCH --requeue

TMAX=100000
STEPSIZE=11
NEXP=12
V=0.5
TOPDIR=/home/jhass2/jamming/JacobData/Binom/

#(tMax, max_step_size, v, Nexp, topDir, sysID) = sys.argv[1:]
python3 BinomialRW.py $TMAX $STEPSIZE $V $NEXP $TOPDIR $SLURM_ARRAY_TASK_ID