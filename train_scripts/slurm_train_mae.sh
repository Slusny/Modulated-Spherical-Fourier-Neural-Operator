#!/bin/bash
####
#a) Define slurm job parameters
####

#SBATCH --job-name=mae #21gcn

#resources:

#SBATCH --ntasks=1 

##SBATCH --nodes=1

#SBATCH --cpus-per-task=8 # 14 is max for cpu-short
# the job can use and see 4 CPUs (from max 24).
# needet task count -n, maybe there is a better way to specify cores

#SBATCH --partition=2080-galvani#a100-galvani#cpu-galvani#2080-galvani#cpu-long#cpu-short #gpu-v100  #gpu-2080ti #cpu-long

#SBATCH --mem-per-cpu=15G # Per CPU -> Per Core
# the job will need 12GB of memory equally distributed on 4 cpus.(251GB are available in total on one node)

#SBATCH --gres=gpu:1
#the job can use and see 1 GPUs (4 GPUs are available in total on one node) use SBATCH --gres=gpu:1080ti:1 to explicitly demand a Geforce 1080 Ti GPU. Use SBATCH --gres=gpu:A4000:1 to explicitly demand a RTX A4000 GPU

#SBATCH --time=00-22:00
# the maximum time the scripts needs to run
# "minutes:seconds", "hours:minutes:seconds", "days-hours","days-hours:minutes" and "days-hours:minutes:seconds"

#SBATCH --error=/home/goswami/gkd965/jobs/job.mae.%J.err
# write the error output to job.*jobID*.err

#SBATCH --output=/home/goswami/gkd965/jobs/job.mae.%J.out
# write the standard output to job.*jobID*.out

#SBATCH --mail-type=ALL
#write a mail if a job begins, ends, fails, gets requeued or stages out

#SBATCH --mail-user=lennart.slusny@student.uni-tuebingen.de
# your mail address


# batch-size 30 for 2080 , 128 for a100
singularity exec --nv --bind /mnt/qb/goswami/data/era5,/mnt/qb/work2/goswami0/gkd965,/scratch_local /mnt/qb/work2/goswami0/gkd965/sfno_packages8.sif /opt/conda/envs/model/bin/python /home/goswami/gkd965/MasterML/main.py --model mae --train --validation-interval 200 --save-checkpoint-interval 10 --validation-epochs 2 --training-workers 6  --learning-rate 0.0005 --advanced-logging --scheduler CosineAnnealingLR --scheduler-horizon 54055 --loss-fn NormalCRPS --trainingset-start-year 1979 --trainingset-end-year 2016 --validationset-start-year 2016 --validationset-end-year 2018 --batch-size 128 --wandb



echo DONE!
