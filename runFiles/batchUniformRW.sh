#!/bin/bash
#SBATCH --job-name=URW50
#SBATCH --time=2-00:00:00
#SBATCH --error=/home/jhass2/jamming/JacobData/logs/Uniform/%A-%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-500
#SBATCH --output=/home/jhass2/jamming/JacobData/logs/Uniform/%A-%a.out
#SBATCH --account=jamming
#SBATCH --partition=preempt
#SBATCH --requeue

TMAX=100000
STEPSIZE=50
NEXP=10
TOPDIR=/home/jhass2/jamming/JacobData/UniformRWMultSteps/$STEPSIZE

mkdir -p $TOPDIR

#(tMax, max_step_size, Nexp, topDir, sysID) = sys.argv[1:]
python3 UniformRW.py $TMAX $STEPSIZE $NEXP $TOPDIR $SLURM_ARRAY_TASK_ID
