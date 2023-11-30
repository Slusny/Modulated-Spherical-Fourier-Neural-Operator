#!/bin/bash
####
#a) Define slurm job parameters
####

#SBATCH --job-name=sfno

#resources:

#SBATCH --ntasks=1 

##SBATCH --nodes=1

#SBATCH --cpus-per-task=12
# the job can use and see 4 CPUs (from max 24).
# needet task count -n, maybe there is a better way to specify cores

#SBATCH --partition=cpu-short
# the slurm partition the job is queued to.
# exits: gpu-2080ti , gpu-v100 ... see sinfo

#SBATCH --mem-per-cpu=64G
# the job will need 12GB of memory equally distributed on 4 cpus.(251GB are available in total on one node)

# #SBATCH --gres=gpu:0
#the job can use and see 1 GPUs (4 GPUs are available in total on one node) use SBATCH --gres=gpu:1080ti:1 to explicitly demand a Geforce 1080 Ti GPU. Use SBATCH --gres=gpu:A4000:1 to explicitly demand a RTX A4000 GPU

#SBATCH --time=0-05:00
# the maximum time the scripts needs to run
# "minutes:seconds", "hours:minutes:seconds", "days-hours","days-hours:minutes" and "days-hours:minutes:seconds"

#SBATCH --error=/home/goswami/gkd965/jobs/job.%J.err
# write the error output to job.*jobID*.err

#SBATCH --output=/home/goswami/gkd965/jobs/job.%J.out
# write the standard output to job.*jobID*.out

#SBATCH --mail-type=ALL
#write a mail if a job begins, ends, fails, gets requeued or stages out

#SBATCH --mail-user=lennart.slusny@student.uni-tuebingen.de
# your mail address

####
#b) copy all needed data to the jobs scratch folder
# We copy the cifar10 datasets which is already available in common datasets folder to our job’s scratch folder.
# Note: For this script, cifar-10 would be downloaded directly from internet, if you didn't copy it yourself.
#### 

# cp -R /common/datasets/cifar_tutorial/ /scratch/$SLURM_JOB_ID/

####
#c) Execute your code in a specific singularity container
#d) Write your checkpoints to your home directory, so that you still have them if your job fails
####

singularity exec --nv --bind /mnt/qb/goswami/data/era5,/mnt/qb/work2/goswami0/gkd965 /mnt/qb/work2/goswami0/gkd965/sfno_packages2.sif /opt/conda/envs/model/bin/python test.py

## juptyer server
# srun --pty bash
# hostname
# singularity shell --bind /mnt/qb/goswami/data/era5,/mnt/qb/work2/goswami0/gkd965 /mnt/qb/work2/goswami0/gkd965/sfno_packages3.sif
# /opt/conda/envs/model/bin/jupyter lab
## new terminal
# ssh -AtL $B_PORT:localhost:$B_PORT gkd965@134.2.168.72 "ssh -AtL $B_PORT:localhost:8888 gkd965@bg-slurmb-bm-3 bash"
# don't use the "novalocal" in the hostname, eg. bg-slurmb-bm-3.novalocalssh -AtL $B_PORT:localhost:$B_PORT gkd965@134.2.168.72 "ssh -AtL $B_PORT:localhost:8888 gkd965@bg-slurmb-bm-3 bash"

echo DONE!
