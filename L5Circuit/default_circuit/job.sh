#!/bin/bash --login

#SBATCH --nodes=10
#SBATCH --ntasks-per-node=40
#SBATCH --time=0:25:00
#SBATCH --job-name='LFPy Circuit'
#SBATCH --account=rrg-etayhay
#SBATCH --mail-type=ALL
#SBATCH --mail-user=agmccrei@gmail.com
#SBATCH -o output.out
#SBATCH -e error.out

module load NiaEnv/2018a
module load intel/2018.2
module load intelmpi/2018.2
module load anaconda3/2018.12

source activate lfpy

unset DISPLAY

mpiexec -n 400 python circuit.py 1234 1
