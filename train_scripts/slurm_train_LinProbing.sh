#!/bin/bash
####
#a) Define slurm job parameters
####

#SBATCH --job-name=linprob #21gcn

#resources:

#SBATCH --ntasks=1 

##SBATCH --nodes=1

#SBATCH --cpus-per-task=8 # 14 is max for cpu-short
# the job can use and see 4 CPUs (from max 24).
# needet task count -n, maybe there is a better way to specify cores

#SBATCH --partition=a100-galvani#2080-galvani#a100-galvani#cpu-galvani#2080-galvani#cpu-long#cpu-short #gpu-v100  #gpu-2080ti #cpu-long

#SBATCH --mem-per-cpu=15G # Per CPU -> Per Core
# the job will need 12GB of memory equally distributed on 4 cpus.(251GB are available in total on one node)

#SBATCH --gres=gpu:1
#the job can use and see 1 GPUs (4 GPUs are available in total on one node) use SBATCH --gres=gpu:1080ti:1 to explicitly demand a Geforce 1080 Ti GPU. Use SBATCH --gres=gpu:A4000:1 to explicitly demand a RTX A4000 GPU

#SBATCH --time=00-22:00
# the maximum time the scripts needs to run
# "minutes:seconds", "hours:minutes:seconds", "days-hours","days-hours:minutes" and "days-hours:minutes:seconds"

#SBATCH --error=/home/goswami/gkd965/jobs/job.linprob.%J.err
# write the error output to job.*jobID*.err

#SBATCH --output=/home/goswami/gkd965/jobs/job.linprob.%J.out
# write the standard output to job.*jobID*.out

#SBATCH --mail-type=ALL
#write a mail if a job begins, ends, fails, gets requeued or stages out

#SBATCH --mail-user=lennart.slusny@student.uni-tuebingen.de
# your mail address


# batch-size 30 for 2080 , 115 for a100 with 28,9,9 ,  16 maybe 20 with 14,9,9
singularity exec --nv --bind /mnt/qb/goswami/data/era5,/mnt/qb/work2/goswami0/gkd965,/home/scratch_local /mnt/qb/work2/goswami0/gkd965/sfno_packages8.sif /opt/conda/envs/model/bin/python /home/goswami/gkd965/MasterML/main.py --model mae --model-version lin-prob --film-gen mae --train --cls /mnt/qb/work2/goswami0/gkd965/checkpoints/mae/wise-spaceship-24/checkpoint_mae_latest_None_iter=0_epoch=8-cls_decoder-1979-2019.npy --advanced-logging --validation-interval 100 --validation-epochs 5 --save-checkpoint-interval 100 --training-workers 6  --learning-rate 0.00005 --scheduler CosineAnnealingLR --scheduler-horizon 270275 --loss-fn MAE --trainingset-start-year 1979 --trainingset-end-year 2016 --validationset-start-year 2016 --validationset-end-year 2018 --training-epochs 5 --batch-size 1 --notes 'decoder cls' --wandb --jobID $SLURM_JOB_ID 

# ssh
# python main.py --model mae --train --validation-interval 20 --save-checkpoint-interval 8 --validation-epochs 2 --training-workers 6  --learning-rate 0.0005 --advanced-logging --scheduler CosineAnnealingLR --scheduler-horizon 432442 --loss-fn NormalCRPS --trainingset-start-year 1979 --trainingset-end-year 2016 --validationset-start-year 2016 --validationset-end-year 2018 --training-epochs 8 --batch-size 64 --patch-size 7 15 30 --wandb --jobID '71530' > /home/goswami/gkd965/jobs/job.mae.71530.out 2> /home/goswami/gkd965/jobs/job.mae.71530.err


echo DONE!

