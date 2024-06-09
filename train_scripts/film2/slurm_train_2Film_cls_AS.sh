#!/bin/bash
####
#a) Define slurm job parameters
####

#SBATCH --job-name=2fdASe #21gcn

#resources:

#SBATCH --ntasks=1 

##SBATCH --nodes=1

#SBATCH --cpus-per-task=60 # 14 is max for cpu-short
# the job can use and see 4 CPUs (from max 24).
# needet task count -n, maybe there is a better way to specify cores

#SBATCH --partition=a100-galvani#2080-galvani#a100-galvani#cpu-galvani#2080-galvani#cpu-long#cpu-short #gpu-v100  #gpu-2080ti #cpu-long

#SBATCH --mem-per-cpu=15G # Per CPU -> Per Core
# the job will need 12GB of memory equally distributed on 4 cpus.(251GB are available in total on one node)

#SBATCH --gres=gpu:8
#the job can use and see 1 GPUs (4 GPUs are available in total on one node) use SBATCH --gres=gpu:1080ti:1 to explicitly demand a Geforce 1080 Ti GPU. Use SBATCH --gres=gpu:A4000:1 to explicitly demand a RTX A4000 GPU

#SBATCH --time=03-00:00
# the maximum time the scripts needs to run
# "minutes:seconds", "hours:minutes:seconds", "days-hours","days-hours:minutes" and "days-hours:minutes:seconds"

#SBATCH --error=/home/goswami/gkd965/jobs/job.2filmCLS.ASe.ddp.%J.err
# write the error output to job.*jobID*.err

#SBATCH --output=/home/goswami/gkd965/jobs/job.2filmCLS.ASe.ddp.%J.out
# write the standard output to job.*jobID*.out

#SBATCH --mail-type=ALL
#write a mail if a job begins, ends, fails, gets requeued or stages out

#SBATCH --mail-user=lennart.slusny@student.uni-tuebingen.de
# your mail address


# batch-size 30 for 2080 , 115 for a100 with 28,9,9 ,  16 maybe 20 with 14,9,9


# 1Film size multi-step-training 20, mit ddp 19 mit ddp
# 8 gpus mit 400MB residual and accumulation step: 19 zu viel, 18 geht? 16 ging

# 2Film size multi-step-training 11, mit ddp 10 mit ddp
# 8 gpus 10 ging nicht, 8 ging

# 2Film
singularity exec --nv --bind /mnt/qb/goswami/data/era5,/mnt/qb/work2/goswami0/gkd965,/home/scratch_local /mnt/qb/work2/goswami0/gkd965/sfno_packages8.sif /opt/conda/envs/model/bin/python /home/goswami/gkd965/MasterML/main.py --model sfno --model-version film --film-gen mae --train --cls /mnt/qb/work2/goswami0/gkd965/checkpoints/mae/apricote-smoke-15/checkpoint_mae_latest_None_iter=0_epoch=8-cls_encoder-1979-2019.npy --advanced-logging --film-layers 2 --batch-size 1 --multi-step-training 2 --training-step-skip 4 --validation-interval 3 --validation-epochs 2 --multi-step-validation 2 --validation-step-skip 15 --save-checkpoint-interval 3 --training-workers 6  --learning-rate 0.00005 --scheduler CosineAnnealingLR --scheduler-horizon 270275 --loss-fn L2Sphere --trainingset-start-year 1979 --trainingset-end-year 2016 --validationset-start-year 2016 --validationset-end-year 2018 --training-epochs 5  --notes 'encoder cls, future sst,Apricote-Smoke-15' --checkpointing-decoder --enable-amp --accumulation-steps 16 --ddp --wandb --jobID $SLURM_JOB_ID 

#ssh 2 Film DDP
# python /home/goswami/gkd965/MasterML/main.py --model sfno --model-version film --film-gen mae --train --cls /mnt/qb/work2/goswami0/gkd965/checkpoints/mae/apricote-smoke-15/checkpoint_mae_latest_None_iter=0_epoch=8-cls_encoder-1979-2019.npy --advanced-logging --film-layers 2 --batch-size 1 --multi-step-training 2 --training-step-skip 4 --validation-interval 3 --validation-epochs 2 --multi-step-validation 2 --validation-step-skip 15 --save-checkpoint-interval 3 --training-workers 6  --learning-rate 0.00005 --scheduler CosineAnnealingLR --scheduler-horizon 270275 --loss-fn L2Sphere --trainingset-start-year 1979 --trainingset-end-year 2016 --validationset-start-year 2016 --validationset-end-year 2018 --training-epochs 5  --notes 'encoder cls, future sst,Apricote-Smoke-15' --checkpointing-decoder --enable-amp --accumulation-steps 16 --ddp --wandb --jobID '1212' > /home/goswami/gkd965/jobs/job.2Film.ASe.ddp.1212.out 2> /home/goswami/gkd965/jobs/job.2Film.ASe.ddp.1212.err

# ssh
# python /home/goswami/gkd965/MasterML/main.py --model mae --train --validation-interval 20 --save-checkpoint-interval 8 --validation-epochs 2 --training-workers 6  --learning-rate 0.0005 --advanced-logging --scheduler CosineAnnealingLR --scheduler-horizon 432442 --loss-fn NormalCRPS --trainingset-start-year 1979 --trainingset-end-year 2016 --validationset-start-year 2016 --validationset-end-year 2018 --training-epochs 8 --batch-size 64 --patch-size 7 15 30 --wandb --jobID '71530' > /home/goswami/gkd965/jobs/job.mae.71530.out 2> /home/goswami/gkd965/jobs/job.mae.71530.err

# jup
# main.py --model sfno --model-version film --film-gen mae --train --cls /mnt/qb/work2/goswami0/gkd965/checkpoints/mae/wise-spaceship-24/checkpoint_mae_latest_None_iter=0_epoch=8-cls_encoder-1979-2019.npy --advanced-logging --film-layers 1 --batch-size 1 --multi-step-training 10 --validation-interval 2 --validation-epochs 1 --multi-step-validation 2 --validation-step-skip 2 --save-checkpoint-interval 1 --training-workers 8  --learning-rate 0.00005 --scheduler CosineAnnealingLR --scheduler-horizon 270275 --loss-fn L2Sphere --trainingset-start-year 1979 --trainingset-end-year 2016 --validationset-start-year 2016 --validationset-end-year 2018 --training-epochs 5  --notes 'decoder cls, future sst' --checkpointing-decoder --enable-amp --accumulation-steps 2 --ddp

echo DONE!

