#!/bin/bash
#SBATCH --job-name=example_slurm_submission                  # create a short name for your job
#SBATCH --nodes=1                                            # node count
#SBATCH --mem-per-cpu=6000                                   # total memory per node 
#SBATCH --gpus=4                                             # number of gpus per node
#SBATCH --cpus-per-task=8                                    # number of CPUs per task
#SBATCH --ntasks-per-node=4                                  # number of tasks per node
#SBATCH --gres=gpumem:24GB                                   # GPU memory
#SBATCH --time=24:00:00                                      # total run time limit (HH:MM:SS)
#SBATCH --output=log_files/example_slurm_submission.out      # standard output file
#SBATCH --error=log_files/example_slurm_submission%a_%A.err  # standard error file
#SBATCH --array=0-4                                          # job array

export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export WORLD_SIZE=4
echo "WORLD_SIZE="$WORLD_SIZE
echo "MASTER_PORT="$MASTER_PORT

export NCCL_SOCKET_IFNAME=lo
export NCCL_DEBUG=INFO

echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr

export PARAM_FOLDER="example_slurm_submission"

# Initialize conda (force initialization if necessary)
conda init bash
source ~/.bashrc

# Activate environment
conda activate rbd_env

# if directory log_files/binary_substrate_only does not exist, create it
if [ ! -d log_files ]; then
    mkdir -p log_files
fi

#ifconfig
python -u main.py @param_files/$PARAM_FOLDER/$PARAM_FILE_QUEUE/$SLURM_ARRAY_TASK_ID.txt
