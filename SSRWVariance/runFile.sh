#!/bin/bash
#SBATCH --job-name=SSRW
#SBATCH --time=7-00:00:00
#SBATCH --error=/home/jhass2/jamming/JacobData/logs/SSRWFPT/%A-%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --array=0
#SBATCH --output=/home/jhass2/jamming/JacobData/logs/SSRWFPT/%A-%a.out
#SBATCH --account=jamming
#SBATCH --partition=preempt
#SBATCH --requeue

for NEXP in 1 2 5 12 28
do
  TOPDIR=/home/jhass2/jamming/JacobData/SSRWFPT/$NEXP
  mkdir -p $TOPDIR
  python3 /home/jhass2/Code/extremeDiffusion1D/SSRWVariance/SSRWFirstPassageTimeNoApprox.py $TOPDIR $NEXP
done
