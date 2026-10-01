#!/bin/bash --login

#SBATCH --nodes=3
#SBATCH --ntasks-per-node=192
#SBATCH --time=0:15:01
#SBATCH --job-name='LFPy Circuit'
#SBATCH --account=rrg-etayhay
#SBATCH --mail-type=ALL
#SBATCH --mail-user=agmccrei@gmail.com
#SBATCH -o output_regular.out
#SBATCH -e error_regular.out

module load StdEnv/2023
module load gcc/13.3
module load openmpi/5.0.3
module load cmake/3.31.0
module load python/3.10.13
module load mpi4py/4.0.0

source $HOME/.virtualenvs/neuron9_bluerecording/bin/activate

export PATH=$HOME/install_neuron9/bin:$PATH
export PYTHONPATH=$HOME/install_neuron9/lib/python:$PYTHONPATH

unset DISPLAY

mpiexec -n 576 python circuit.py 1234 1234
