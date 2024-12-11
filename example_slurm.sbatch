#!/bin/bash
####
# An example SLURM script to evaluate the standard sfno model.
# Adapt the script to your needs.
####

#SBATCH --job-name=gcn
# the name of the job

#resources:

#SBATCH --ntasks=1 

##SBATCH --nodes=1

#SBATCH --cpus-per-task=25 
# number of cores to allocate

#SBATCH --partition=gpu-v100 

#SBATCH --mem-per-cpu=25G # Memory Per Core

#SBATCH --gres=gpu:8
# number of GPUs to allocate

#SBATCH --time=03-00:00
# the maximum time the scripts needs to run
# "minutes:seconds", "hours:minutes:seconds", "days-hours","days-hours:minutes" and "days-hours:minutes:seconds"

#SBATCH --error=/home/goswami/gkd965/jobs/GCN.%J.err
# write the error output to job.*jobID*.err

#SBATCH --output=/home/goswami/gkd965/jobs/GCN.%J.out
# write the standard output to job.*jobID*.out

#SBATCH --mail-type=ALL
# write a mail if a job begins, ends, fails, gets requeued or stages out

#SBATCH --mail-user=lennart.slusny@student.uni-tuebingen.de
# your mail address

# run the configuration you want to use
# this example uses a singularity container as an environment for the script. Use --nv to enable GPU support and --bind to mount directories
singularity exec --nv --bind /mnt/qb/goswami/data/era5,/mnt/qb/work2/goswami0/gkd965 /mnt/qb/work2/goswami0/gkd965/sfno_packages8.sif /opt/conda/envs/model/bin/python main.py --model sfno --test --training-workers 3 --batch-size 3
