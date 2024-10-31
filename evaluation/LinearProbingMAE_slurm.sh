#!/bin/bash
####
#a) Define slurm job parameters
####

#SBATCH --job-name=lpeval #21gcn

#resources:

#SBATCH --ntasks=1 

##SBATCH --nodes=1

#SBATCH --cpus-per-task=8 

#SBATCH --partition=a100-galvani

#SBATCH --mem-per-cpu=8G # 

#SBATCH --gres=gpu:1

#SBATCH --time=00-22:00
# the maximum time the scripts needs to run
# "minutes:seconds", "hours:minutes:seconds", "days-hours","days-hours:minutes" and "days-hours:minutes:seconds"

#SBATCH --error=/home/goswami/gkd965/jobs/job.eval.linprob.%J.err
# write the error output to job.*jobID*.err

#SBATCH --output=/home/goswami/gkd965/jobs/job.eval.linprob.%J.out
# write the standard output to job.*jobID*.out

#SBATCH --mail-type=ALL
#write a mail if a job begins, ends, fails, gets requeued or stages out

#SBATCH --mail-user=lennart.slusny@student.uni-tuebingen.de
# your mail address

#GDe
python /home/goswami/gkd965/MasterML/main.py --model mae --model-version lin-probe --run --cls /mnt/qb/work2/goswami0/gkd965/checkpoints/mae/glorious-deluge-23/checkpoint_mae_latest_None_iter=0_epoch=8-cls_encoder-1979-2019.npy --oni-path /mnt/qb/work2/goswami0/gkd965/Assets/mae/oni.npy --advanced-logging --training-workers 7 --loss-fn MSE --trainingset-start-year 1979 --trainingset-end-year 2016 --validationset-start-year 2016 --validationset-end-year 2018 --batch-size 1000 --resume-checkpoint /mnt/qb/work2/goswami0/gkd965/checkpoints/rose-firebrand-42-sID{398100}/checkpoint_mae_lin-probe_mae_iter=0_epoch=60.pkl
#lr up-down
python /home/goswami/gkd965/MasterML/main.py --model mae --model-version lin-probe --run --cls /mnt/qb/work2/goswami0/gkd965/checkpoints/mae/glorious-deluge-23/checkpoint_mae_latest_None_iter=0_epoch=8-cls_encoder-1979-2019.npy --oni-path /mnt/qb/work2/goswami0/gkd965/Assets/mae/oni.npy --advanced-logging --training-workers 7 --loss-fn MSE --trainingset-start-year 1979 --trainingset-end-year 2016 --validationset-start-year 2016 --validationset-end-year 2018 --batch-size 1000 --resume-checkpoint /mnt/qb/work2/goswami0/gkd965/checkpoints/tmp/comic-firebrand-35-sID{398081}/checkpoint_mae_lin-probe_mae_iter=0_epoch=60.pkl

#GDd
python /home/goswami/gkd965/MasterML/main.py --model mae --model-version lin-probe --run --cls /mnt/qb/work2/goswami0/gkd965/checkpoints/mae/glorious-deluge-23/checkpoint_mae_latest_None_iter=0_epoch=8-cls_decoder-1979-2019.npy --oni-path /mnt/qb/work2/goswami0/gkd965/Assets/mae/oni.npy --advanced-logging --training-workers 7 --loss-fn MSE --trainingset-start-year 1979 --trainingset-end-year 2016 --validationset-start-year 2016 --validationset-end-year 2018 --batch-size 1000 --resume-checkpoint /mnt/qb/work2/goswami0/gkd965/checkpoints/kind-music-43-sID{398101}/checkpoint_mae_lin-probe_mae_iter=0_epoch=60.pkl

#WSe
python /home/goswami/gkd965/MasterML/main.py --model mae --model-version lin-probe --run --cls /mnt/qb/work2/goswami0/gkd965/checkpoints/mae/wise-spaceship-24/checkpoint_mae_latest_None_iter=0_epoch=8-cls_encoder-1979-2019.npy --oni-path /mnt/qb/work2/goswami0/gkd965/Assets/mae/oni.npy --advanced-logging --training-workers 7 --loss-fn MSE --trainingset-start-year 1979 --trainingset-end-year 2016 --validationset-start-year 2016 --validationset-end-year 2018 --batch-size 1000 --resume-checkpoint /mnt/qb/work2/goswami0/gkd965/checkpoints/chocolate-monkey-31-sID{398071}/checkpoint_mae_lin-probe_mae_iter=0_epoch=60.pkl #earnest-shape-28-sID{397404}  45

#WSd
python /home/goswami/gkd965/MasterML/main.py --model mae --model-version lin-probe --run --cls /mnt/qb/work2/goswami0/gkd965/checkpoints/mae/wise-spaceship-24/checkpoint_mae_latest_None_iter=0_epoch=8-cls_decoder-1979-2019.npy --oni-path /mnt/qb/work2/goswami0/gkd965/Assets/mae/oni.npy --advanced-logging --training-workers 7 --loss-fn MSE --trainingset-start-year 1979 --trainingset-end-year 2016 --validationset-start-year 2016 --validationset-end-year 2018 --batch-size 1000 --resume-checkpoint /mnt/qb/work2/goswami0/gkd965/checkpoints/swift-thunder-32-sID{398078}/checkpoint_mae_lin-probe_mae_iter=0_epoch=60.pkl #denim-sponge-15-sID{396677}

#HSe
python /home/goswami/gkd965/MasterML/main.py --model mae --model-version lin-probe --run --cls /mnt/qb/work2/goswami0/gkd965/checkpoints/mae/helpful-salad-14/checkpoint_mae_latest_None_iter=0_epoch=8-cls_encoder-1979-2019.npy --oni-path /mnt/qb/work2/goswami0/gkd965/Assets/mae/oni.npy --advanced-logging --training-workers 7 --loss-fn MSE --trainingset-start-year 1979 --trainingset-end-year 2016 --validationset-start-year 2016 --validationset-end-year 2018 --batch-size 1000 --resume-checkpoint /mnt/qb/work2/goswami0/gkd965/checkpoints/lilac-puddle-41-sID{398099}/checkpoint_mae_lin-probe_mae_iter=0_epoch=60.pkl

#HSd
python /home/goswami/gkd965/MasterML/main.py --model mae --model-version lin-probe --run --cls /mnt/qb/work2/goswami0/gkd965/checkpoints/mae/helpful-salad-14/checkpoint_mae_latest_None_iter=0_epoch=8-cls_decoder-1979-2019.npy --oni-path /mnt/qb/work2/goswami0/gkd965/Assets/mae/oni.npy --advanced-logging --training-workers 7 --loss-fn MSE --trainingset-start-year 1979 --trainingset-end-year 2016 --validationset-start-year 2016 --validationset-end-year 2018 --batch-size 1000 --resume-checkpoint  /mnt/qb/work2/goswami0/gkd965/checkpoints/fresh-deluge-40-sID{398098}/checkpoint_mae_lin-probe_mae_iter=0_epoch=60.pkl

#ASe
python /home/goswami/gkd965/MasterML/main.py --model mae --model-version lin-probe --run --cls /mnt/qb/work2/goswami0/gkd965/checkpoints/mae/apricot-smoke-15/checkpoint_mae_latest_None_iter=0_epoch=8-cls_encoder-1979-2019.npy --oni-path /mnt/qb/work2/goswami0/gkd965/Assets/mae/oni.npy --advanced-logging --training-workers 7 --loss-fn MSE --trainingset-start-year 1979 --trainingset-end-year 2016 --validationset-start-year 2016 --validationset-end-year 2018 --batch-size 1000 --resume-checkpoint /mnt/qb/work2/goswami0/gkd965/checkpoints/peachy-energy-20-sID{397369}/checkpoint_mae_lin-probe_mae_iter=0_epoch=14.pkl  
#new
python /home/goswami/gkd965/MasterML/main.py --model mae --model-version lin-probe --run --cls /mnt/qb/work2/goswami0/gkd965/checkpoints/mae/apricot-smoke-15/checkpoint_mae_latest_None_iter=0_epoch=8-cls_encoder-1979-2019.npy --oni-path /mnt/qb/work2/goswami0/gkd965/Assets/mae/oni.npy --advanced-logging --training-workers 7 --loss-fn MSE --trainingset-start-year 1979 --trainingset-end-year 2016 --validationset-start-year 2016 --validationset-end-year 2018 --batch-size 1000 --resume-checkpoint /mnt/qb/work2/goswami0/gkd965/checkpoints/bumbling-field-44-sID{398102}/checkpoint_mae_lin-probe_mae_iter=0_epoch=60.pkl 

#ASd
python /home/goswami/gkd965/MasterML/main.py --model mae --model-version lin-probe --run --cls /mnt/qb/work2/goswami0/gkd965/checkpoints/mae/apricot-smoke-15/checkpoint_mae_latest_None_iter=0_epoch=8-cls_decoder-1979-2019.npy --oni-path /mnt/qb/work2/goswami0/gkd965/Assets/mae/oni.npy --advanced-logging --training-workers 7 --loss-fn MSE --trainingset-start-year 1979 --trainingset-end-year 2016 --validationset-start-year 2016 --validationset-end-year 2018 --batch-size 1000 --resume-checkpoint /mnt/qb/work2/goswami0/gkd965/checkpoints/misty-silence-37-sID{398083}/checkpoint_mae_lin-probe_mae_iter=0_epoch=60.pkl 

#Ones
python /home/goswami/gkd965/MasterML/main.py --model mae --model-version lin-probe --run --cls /mnt/qb/work2/goswami0/gkd965/Assets/mae/ones_cls.npy --oni-path /mnt/qb/work2/goswami0/gkd965/Assets/mae/oni.npy --advanced-logging --training-workers 7 --loss-fn MSE --trainingset-start-year 1979 --trainingset-end-year 2016 --validationset-start-year 2016 --validationset-end-year 2018 --batch-size 1000 --resume-checkpoint /mnt/qb/work2/goswami0/gkd965/checkpoints/glamorous-breeze-39-sID{398096}/checkpoint_mae_lin-probe_mae_iter=0_epoch=60.pkl

echo DONE!

