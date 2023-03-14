#!/bin/bash

# Usage: sbatch ~/scripts/run.sh <job_script> (<parameters>)*

#SBATCH -N 1 # number nodes
#SBATCH --exclude=charles11,charles12,charles13,charles14 #,apollo1
#SBATCH --partition=cdtgpucluster,apollo
#SBATCH -n 1 # number tasks (cpu)
#SBATCH --time=0-6:00:00 # previously 12:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=14000
#SBATCH --output="/home/s1569253/stdout/base_exp_%j.out"
#SBATCH --error="/home/s1569253/stdout/base_exp_%j.err"

echo "CUDA_VISIBLE_DEVICES = ${CUDA_VISIBLE_DEVICES}"
echo "hostname = $(hostname)"

nvidia-smi

export ENV_HOME=$HOME/miniconda/py38

source ~/.bashrc

job_script=$1
params=${@:1}

# activate the python environment
source activate py38
export MPLBACKEND="agg"

# Run the executable
#python_executable=$(mktemp)
#echo "#!/bin/bash" >> $python_executable
#echo "python -u $job_script $params" >> $python_executable

echo "${params}"

sh run_model_bin.sh $params

