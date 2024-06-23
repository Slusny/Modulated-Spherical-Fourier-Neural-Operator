#!/bin/bash
####
#a) Define slurm job parameters
####

#SBATCH --job-name=21gcn #21gcn

#resources:

#SBATCH --ntasks=1 

##SBATCH --nodes=1

#SBATCH --cpus-per-task=8 # 14 is max for cpu-short
# the job can use and see 4 CPUs (from max 24).
# needet task count -n, maybe there is a better way to specify cores

#SBATCH --partition=a100-galvani#a100-galvani#cpu-galvani#2080-galvani#cpu-long#cpu-short #gpu-v100  #gpu-2080ti #cpu-long

#SBATCH --mem-per-cpu=15G # Per CPU -> Per Core
# the job will need 12GB of memory equally distributed on 4 cpus.(251GB are available in total on one node)

#SBATCH --gres=gpu:1
#the job can use and see 1 GPUs (4 GPUs are available in total on one node) use SBATCH --gres=gpu:1080ti:1 to explicitly demand a Geforce 1080 Ti GPU. Use SBATCH --gres=gpu:A4000:1 to explicitly demand a RTX A4000 GPU

#SBATCH --time=00-22:00
# the maximum time the scripts needs to run
# "minutes:seconds", "hours:minutes:seconds", "days-hours","days-hours:minutes" and "days-hours:minutes:seconds"

#SBATCH --error=/home/goswami/gkd965/jobs/job.21gcn.%J.err
# write the error output to job.*jobID*.err

#SBATCH --output=/home/goswami/gkd965/jobs/job.21gcn.%J.out
# write the standard output to job.*jobID*.out

#SBATCH --mail-type=ALL
#write a mail if a job begins, ends, fails, gets requeued or stages out

#SBATCH --mail-user=lennart.slusny@student.uni-tuebingen.de
# your mail address

singularity exec --nv --bind /mnt/qb/goswami/data/era5,/mnt/qb/work2/goswami0/gkd965 /mnt/qb/work2/goswami0/gkd965/sfno_packages8.sif /opt/conda/envs/model/bin/python /home/goswami/gkd965/MasterML/main.py --model sfno --model-version film --train --validation-interval 100 --save-checkpoint-interval 5 --validation-epochs 5 --training-workers 6 --batch-size 1 --learning-rate 0.0005 --multi-step-training 2 --training-step-skip 1 --film-gen gcn --wandb --advanced-logging -checkpointing-block --checkpointing-encoder 

# ssh
python /home/goswami/gkd965/MasterML/main.py --train --model sfno --model-version film --film-gen vit --training-workers 6 --multi-step-training 3 --advanced-logging --model-depth 6 --film-layers 2 --learning-rate 0.0005 --discount-factor 0.85 --ddp --scheduler CosineAnnealingLR --scheduler-horizon 108110 --loss-fn L2Sphere_noSine --validation-interval 6 --validation-epochs 2 --multi-step-validation 2 --validation-step-skip 15 --save-checkpoint-interval 20 --training-epochs 2 --accumulation-steps 16 --enable-amp --checkpointing-decoder --retrain-film --wandb --wandb-project gcn --jobID 21191 > /home/goswami/gkd965/jobs/job.gcn.retrain.21191.out 2> /home/goswami/gkd965/jobs/job.gcn.resume.21191.out


echo DONE!

