#!/bin/bash
####
#a) Define slurm job parameters
####

#SBATCH --job-name=sfno

#resources:

#SBATCH --ntasks=1 

##SBATCH --nodes=1

#SBATCH --cpus-per-task=6 # 14 is max for cpu-short
# the job can use and see 4 CPUs (from max 24).
# needet task count -n, maybe there is a better way to specify cores

#SBATCH --partition=a100-galvani#a100-galvani#cpu-galvani#2080-galvani#cpu-long#cpu-short #gpu-v100  #gpu-2080ti #cpu-long

#SBATCH --mem-per-cpu=30G # Per CPU -> Per Core
# the job will need 12GB of memory equally distributed on 4 cpus.(251GB are available in total on one node)

#SBATCH --gres=gpu:1
#the job can use and see 1 GPUs (4 GPUs are available in total on one node) use SBATCH --gres=gpu:1080ti:1 to explicitly demand a Geforce 1080 Ti GPU. Use SBATCH --gres=gpu:A4000:1 to explicitly demand a RTX A4000 GPU

#SBATCH --time=01-01:00
# the maximum time the scripts needs to run
# "minutes:seconds", "hours:minutes:seconds", "days-hours","days-hours:minutes" and "days-hours:minutes:seconds"

#SBATCH --error=/home/goswami/gkd965/jobs/job.sfno.%J.err
# write the error output to job.*jobID*.err

#SBATCH --output=/home/goswami/gkd965/jobs/job.sfno.%J.out
# write the standard output to job.*jobID*.out

#SBATCH --mail-type=ALL
#write a mail if a job begins, ends, fails, gets requeued or stages out

#SBATCH --mail-user=lennart.slusny@student.uni-tuebingen.de
# your mail address

singularity exec --nv --bind /mnt/qb/goswami/data/era5,/mnt/qb/work2/goswami0/gkd965 /mnt/qb/work2/goswami0/gkd965/sfno_packages8.sif /opt/conda/envs/model/bin/python /home/goswami/gkd965/MasterML/main.py --model sfno --model-version latest --train --validation-interval 20 --validation-epochs 5 --training-workers 6 --batch-size 1 --wandb --notes "scheduler cosinewarmup t_0=200, ADAM,"

echo DONE!

