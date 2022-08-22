
#!/bin/bash
#SBATCH --job-name=FCDF
#SBATCH --time=1-00:00:00
#SBATCH --error=/home/jhass2/jamming/JacobData/logs/FixedFirstPassCDF/%A-%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-499
#SBATCH --output=/home/jhass2/jamming/JacobData/logs/FixedFirstPassCDF/%A-%a.out
#SBATCH --account=jamming
#SBATCH --partition=preempt

NEXP=24
DMIN=0
DMAX=500
NUMOFPOINTS=500
TOPDIR=/home/jhass2/jamming/JacobData/FixedFirstPassCDF/

mkdir -p $TOPDIR

python3 NumbaFirstPassage.py $TOPDIR $SLURM_ARRAY_TASK_ID $DMIN $DMAX $NEXP $NUMOFPOINTS
