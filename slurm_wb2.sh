#!/bin/bash
####
#a) Define slurm job parameters
####

#SBATCH --job-name=wb2

#resources:

#SBATCH --ntasks=1 

##SBATCH --nodes=1

##SBATCH --cpus-per-task=1 # 14 is max for cpu-short
# the job can use and see 4 CPUs (from max 24).
# needet task count -n, maybe there is a better way to specify cores

#SBATCH --partition=gpu-2080ti#cpu-short#cpu-short #gpu-v100  #gpu-2080ti #cpu-long
# the slurm partition the job is queued to.
# exits: gpu-2080ti , gpu-v100 ... see sinfo

## SBATCH --mem-per-cpu=2G # Per CPU -> Per Core
#SBATCH --mem-per-cpu=2G # Per CPU -> Per Core
# the job will need 12GB of memory equally distributed on 4 cpus.(251GB are available in total on one node)

## SBATCH --gres=gpu:1
#the job can use and see 1 GPUs (4 GPUs are available in total on one node) use SBATCH --gres=gpu:1080ti:1 to explicitly demand a Geforce 1080 Ti GPU. Use SBATCH --gres=gpu:A4000:1 to explicitly demand a RTX A4000 GPU

#SBATCH --time=00-1:00
# the maximum time the scripts needs to run
# "minutes:seconds", "hours:minutes:seconds", "days-hours","days-hours:minutes" and "days-hours:minutes:seconds"

#SBATCH --error=/home/goswami/gkd965/jobs/job.wb2.%J.err
# write the error output to job.*jobID*.err

#SBATCH --output=/home/goswami/gkd965/jobs/job.wb2.%J.out
# write the standard output to job.*jobID*.out

#SBATCH --mail-type=ALL
#write a mail if a job begins, ends, fails, gets requeued or stages out

#SBATCH --mail-user=lennart.slusny@student.uni-tuebingen.de
# your mail address

####
#b) copy all needed data to the jobs scratch folder
# We copy the cifar10 datasets which is already available in common datasets folder to our job’s scratch folder.
# Note: For this script, cifar-10 sfno
#d) Write your checkpoints to your home directory, so that you still have them if your job fails
####

# singularity e xec --nv --bind /mnt/qb/goswami/data/era5,/mnt/qb/work2/goswami0/gkd965 /mnt/qb/work2/goswami0/gkd965/sfno_packages5.sif /opt/conda/envs/model/bin/python main.py --model sfno --date 20210101 --time 0000 --lead-time 120 --path /mnt/qb/work2/goswami0/gkd965/outputs --assets /mnt/qb/work2/goswami0/gkd965/Assets --dump-provenance #8760
#
# singularity exec --nv --bind /mnt/qb/goswami/data/era5,/mnt/qb/work2/goswami0/gkd965 /mnt/qb/work2/goswami0/gkd965/sfno_packages5.sif /opt/conda/envs/model/bin/python main.py --model sfno --date 20190101 --time 0000 --lead-time 8760 --assets /mnt/qb/work2/goswami0/gkd965/Assets --path /mnt/qb/work2/goswami0/gkd965/outputs --dump-provenance --output netcdf --file /mnt/qb/work2/goswami0/gkd965/ClimateInputData_201901010.grib
# singularity exec --nv --bind /mnt/qb/goswami/data/era5,/mnt/qb/work2/goswami0/gkd965 /mnt/qb/work2/goswami0/gkd965/sfno_packages5.sif /opt/conda/envs/model/bin/python convert_to_netcdf.py 

singularity exec --nv --bind /mnt/qb/goswami/data/era5,/mnt/qb/work2/goswami0/gkd965 /mnt/qb/work2/goswami0/gkd965/setup.sif /home/goswami/gkd965/MasterML/download_wb2.sh #parallel_clima_byhand.py 2 #chunky_climatology.py 2
singularity exec --nv --bind /mnt/qb/goswami/data/era5,/mnt/qb/work2/goswami0/gkd965 /mnt/qb/work2/goswami0/gkd965/setup.sif gsutil -m cp -r "gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr" /mnt/qb/goswami/data/era5/weatherbench2/
#parallel_clima_byhand.py 2 #chunky_climatology.py 2


## juptyer server
# srun --pty bash --gres=gpu:1
# hostname
# singularity shell --nv --bind /mnt/qb/goswami/data/era5,/mnt/qb/work2/goswami0/gkd965 /mnt/qb/work2/goswami0/gkd965/sfno_packages5.sif
# /opt/conda/envs/model/bin/jupyter lab
## new terminal
# ssh -AtL $B_PORT:localhost:$B_PORT gkd965@134.2.168.72 "ssh -AtL $B_PORT:localhost:8888 gkd965@bg-slurmb-bm-3 bash"
# don't use the "novalocal" in the hostname, eg. bg-slurmb-bm-3.novalocalssh -AtL $B_PORT:localhost:$B_PORT gkd965@134.2.168.72 "ssh -AtL $B_PORT:localhost:8888 gkd965@bg-slurmb-bm-3 bash"

echo DONE!

#QOS: long-10d-jobs,normal
# Name         |Priority|GraceTime|Preempt|PreemptExemptTime|PreemptMode|Flags                         |UsageFactor|GrpTRES|MaxWall    |MaxTRESPU                                                         |MaxJobsPU|MaxSubmitPU|MaxTRESPA|MaxJobsPA|MaxSubmitPA|MinTRES|
# normal       |0       |00:00:00 |       |                 |cluster    |                              |1.000000   |       |           ||||||||
# long-10d-jobs|0       |00:00:00 |       |                 |cluster    |DenyOnLimit,PartitionTimeLimit|1.000000   |node=12|10-00:00:00|cpu=72,gres/gpu:rtx2080ti=8,gres/gpu:v100=0,gres/gpu=8,mem=354550M|2|20||4|100||

#scontrol show nodes

# NodeName=slurm-bm-06 CoresPerSocket=18 
#    CPUAlloc=0 CPUTot=72 CPULoad=N/A
#    AvailableFeatures=nodata,nvidia_v515,qb_v2.44.1
#    ActiveFeatures=nodata,nvidia_v515,qb_v2.44.1
#    Gres=gpu:rtx2080ti:8
#    NodeAddr=192.168.212.190 NodeHostName=slurm-bm-06 
#    RealMemory=354566 AllocMem=0 FreeMem=N/A Sockets=2 Boards=1
#    State=DOWN*+DRAIN ThreadsPerCore=2 TmpDisk=0 Weight=100 Owner=N/A MCS_label=N/A
#    Partitions=gpu-2080ti,gpu-2080ti-dev,gpu-2080ti-interactive,gpu-2080ti-large,gpu-2080ti-preemptable,service 
#    BootTime=None SlurmdStartTime=None
#    CfgTRES=cpu=72,mem=354566M,billing=40,gres/gpu=8,gres/gpu:rtx2080ti=8
#    AllocTRES=
#    CapWatts=n/a
#    CurrentWatts=0 AveWatts=0
#    ExtSensorsJoules=n/s ExtSensorsWatts=0 ExtSensorsTemp=n/s
#    Reason=Pathfinder [centos@2023-10-25T11:05:35]
#    Comment=(null)


# NodeName=slurm-cpu-hm-1 CoresPerSocket=16
#    CPUAlloc=0 CPUTot=64 CPULoad=N/A
#    AvailableFeatures=nodata,qb_v2.44.1
#    ActiveFeatures=nodata,qb_v2.44.1
#    Gres=(null)
#    NodeAddr=192.168.212.224 NodeHostName=slurm-cpu-hm-1
#    RealMemory=1138967 AllocMem=0 FreeMem=N/A Sockets=2 Boards=1
#    State=DOWN*+DRAIN ThreadsPerCore=2 TmpDisk=0 Weight=100 Owner=N/A MCS_label=N/A
#    Partitions=cpu-long,cpu-preemptable,cpu-short
#    BootTime=None SlurmdStartTime=None
#    CfgTRES=cpu=64,mem=1138967M,billing=14
#    AllocTRES=
#    CapWatts=n/a
#    CurrentWatts=0 AveWatts=0
#    ExtSensorsJoules=n/s ExtSensorsWatts=0 ExtSensorsTemp=n/s
#    Reason=Moved-to-Galvani-CN [centos@2023-11-21T13:31:02]
#    Comment=(null)