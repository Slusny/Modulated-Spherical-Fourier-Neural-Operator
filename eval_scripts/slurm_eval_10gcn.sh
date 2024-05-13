#!/bin/bash
####
#a) Define slurm job parameters
####

#SBATCH --job-name=evalg12l

#resources:

#SBATCH --ntasks=1 

##SBATCH --nodes=1

#SBATCH --cpus-per-task=6 # 14 is max for cpu-short
# the job can use and see 4 CPUs (from max 24).
# needet task count -n, maybe there is a better way to specify cores

#SBATCH --partition=2080-galvani#a100-galvani#cpu-galvani#2080-galvani#cpu-long#cpu-short #gpu-v100  #gpu-2080ti #cpu-long

#SBATCH --mem-per-cpu=25G # Per CPU -> Per Core
# the job will need 12GB of memory equally distributed on 4 cpus.(251GB are available in total on one node)

#SBATCH --gres=gpu:1
#the job can use and see 1 GPUs (4 GPUs are available in total on one node) use SBATCH --gres=gpu:1080ti:1 to explicitly demand a Geforce 1080 Ti GPU. Use SBATCH --gres=gpu:A4000:1 to explicitly demand a RTX A4000 GPU

#SBATCH --time=00-19:00
# the maximum time the scripts needs to run
# "minutes:seconds", "hours:minutes:seconds", "days-hours","days-hours:minutes" and "days-hours:minutes:seconds"

#SBATCH --error=/home/goswami/gkd965/jobs/job.eval10gcn12b.%J.err
# write the error output to job.*jobID*.err

#SBATCH --output=/home/goswami/gkd965/jobs/job.eval10gcn12b.%J.out
# write the standard output to job.*jobID*.out

#SBATCH --mail-type=ALL
#write a mail if a job begins, ends, fails, gets requeued or stages out

#SBATCH --mail-user=lennart.slusny@student.uni-tuebingen.de
# your mail address

singularity exec --nv --bind /mnt/qb/goswami/data/era5,/mnt/qb/work2/goswami0/gkd965 /mnt/qb/work2/goswami0/gkd965/sfno_packages8.sif /opt/conda/envs/model/bin/python /home/goswami/gkd965/MasterML/main.py --validation-epochs 15 --training-workers 7 --batch-size 1 --multi-step-validation 5 --validation-step-skip 5 --eval-model --eval-checkpoint-path /mnt/qb/work2/goswami0/gkd965/checkpoints/sfno_film_gcn_20240512T0427 --eval-checkpoint-num 3  #--eval-checkpoint-num 5 #--eval-checkpoints

#local 
# main.py --validation-epochs 15 --training-workers 7 --batch-size 1 --multi-step-validation 5 --validation-step-skip 5 --eval-model --eval-checkpoint-path /media/lenny/V/Master/checkpoints/apricot-smoke-15 --eval-checkpoint-num 2 --trainingset-start-year 2001 --trainingset-end-year 2002 --validationset-start-year 2002 --validationset-end-year 2003 --trainingdata-path /media/lenny/V/wb2_2001-2003.zarr --assets /mnt/ssd2/Master/S2S_on_SFNO/Assets 

echo DONE!