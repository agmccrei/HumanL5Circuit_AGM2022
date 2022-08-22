#!/bin/bash --login

#SBATCH --nodes=3
#SBATCH --ntasks-per-node=1
#SBATCH --time=0:30:00
#SBATCH --job-name='Neuron Optimization'
#SBATCH --account=rrg-etayhay
#SBATCH --mail-type=ALL
#SBATCH --mail-user=agmccrei@gmail.com
#SBATCH -o output.out
#SBATCH -e error.out
#SBATCH -p debug

module load NiaEnv/2018a
module load intel/2018.2
module load intelmpi/2018.2
module load anaconda3/2018.12

conda activate lfpy
profile=${SLURM_JOB_ID}_$(hostname)

unset DISPLAY

echo "Starting job ${SLURM_JOB_ID}"
ipython profile create --parallel ${profile} --ipython-dir=$SCRATCH/.ipython

echo "Launching controller"
ipcontroller --ip="*" --profile=${profile} --log-to-file --ipython-dir=$SCRATCH/.ipython &
sleep 10

echo "Launching engines"
srun ipengine --profile=${profile} --location=$(hostname) --log-to-file --ipython-dir=$SCRATCH/.ipython &
sleep 45

echo "Launching job"

python init.py --profile ${profile}

if [ $? -eq 0 ]
then
		echo "Job ${SLURM_JOB_ID} completed successfully!"
else
		echo "FAILURE: Job ${SLURM_JOB_ID}"
fi
