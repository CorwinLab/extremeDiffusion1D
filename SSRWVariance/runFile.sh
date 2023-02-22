#!/bin/bash
#SBATCH --job-name=SSRW
#SBATCH --time=7-00:00:00
#SBATCH --error=/home/jhass2/CleanData/logs/SSRWFPT/%A-%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-499
#SBATCH --output=/home/jhass2/CleanData/logs/SSRWFPT/%A-%a.out
#SBATCH --nice=1000

for NEXP in 1 2 5 12 28
do
  TOPDIR=/home/jhass2/CleanData/SSRWFPT/$BETA
  mkdir -p $TOPDIR
  python3 SSRWFiristPassageTimeNoApprox.py $TOPDIR $NEXP
done