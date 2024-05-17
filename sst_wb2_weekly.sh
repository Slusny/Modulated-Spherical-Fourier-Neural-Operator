#!/bin/bash
####
#a) Define slurm job parameters
####

#SBATCH --job-name=wb2

#resources:

#SBATCH --ntasks=1 

##SBATCH --nodes=1

#SBATCH --cpus-per-task=10 # 14 is max for cpu-short
# the job can use and see 4 CPUs (from max 24).
# needet task count -n, maybe there is a better way to specify cores

#SBATCH --partition=2080-galvani#cpu-galvani#2080-galvani#gpu-2080ti#cpu-short#cpu-short #gpu-v100  #gpu-2080ti #cpu-long
# the slurm partition the job is queued to.
# exits: gpu-2080ti , gpu-v100 ... see sinfo

## SBATCH --mem-per-cpu=2G # Per CPU -> Per Core
#SBATCH --mem-per-cpu=15G # Per CPU -> Per Core
# the job will need 12GB of memory equally distributed on 4 cpus.(251GB are available in total on one node)

## SBATCH --gres=gpu:1
#the job can use and see 1 GPUs (4 GPUs are available in total on one node) use SBATCH --gres=gpu:1080ti:1 to explicitly demand a Geforce 1080 Ti GPU. Use SBATCH --gres=gpu:A4000:1 to explicitly demand a RTX A4000 GPU

#SBATCH --time=00-20:00
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

# singularity exec --nv --bind /mnt/qb/goswami/data/era5,/mnt/qb/work2/goswami0/gkd965 /mnt/qb/work2/goswami0/gkd965/setup.sif /home/goswami/gkd965/MasterML/download_wb2.sh #parallel_clima_byhand.py 2 #chunky_climatology.py 2
singularity exec --nv --bind /mnt/qb/goswami/data/era5,/mnt/qb/work2/goswami0/gkd965 /mnt/qb/work2/goswami0/gkd965/setup.sif gsutil -m cp \
  "gs://weatherbench2/datasets/era5_weekly/1959-2023_01_10-full_37-1h-0p25deg-chunk-1-s2s.zarr/sea_surface_temperature/.zarray" \
  "gs://weatherbench2/datasets/era5_weekly/1959-2023_01_10-full_37-1h-0p25deg-chunk-1-s2s.zarr/sea_surface_temperature/.zattrs" \
  "gs://weatherbench2/datasets/era5_weekly/1959-2023_01_10-full_37-1h-0p25deg-chunk-1-s2s.zarr/sea_surface_temperature/0.0.0" \
  "gs://weatherbench2/datasets/era5_weekly/1959-2023_01_10-full_37-1h-0p25deg-chunk-1-s2s.zarr/sea_surface_temperature/1.0.0" \
  "gs://weatherbench2/datasets/era5_weekly/1959-2023_01_10-full_37-1h-0p25deg-chunk-1-s2s.zarr/sea_surface_temperature/10.0.0" \
  "gs://weatherbench2/datasets/era5_weekly/1959-2023_01_10-full_37-1h-0p25deg-chunk-1-s2s.zarr/sea_surface_temperature/100.0.0" \
  "gs://weatherbench2/datasets/era5_weekly/1959-2023_01_10-full_37-1h-0p25deg-chunk-1-s2s.zarr/sea_surface_temperature/1000.0.0" \
  "gs://weatherbench2/datasets/era5_weekly/1959-2023_01_10-full_37-1h-0p25deg-chunk-1-s2s.zarr/sea_surface_temperature/10000.0.0" \
  "gs://weatherbench2/datasets/era5_weekly/1959-2023_01_10-full_37-1h-0p25deg-chunk-1-s2s.zarr/sea_surface_temperature/10001.0.0" \
  "gs://weatherbench2/datasets/era5_weekly/1959-2023_01_10-full_37-1h-0p25deg-chunk-1-s2s.zarr/sea_surface_temperature/10002.0.0" \
  "gs://weatherbench2/datasets/era5_weekly/1959-2023_01_10-full_37-1h-0p25deg-chunk-1-s2s.zarr/sea_surface_temperature/10003.0.0" \
  "gs://weatherbench2/datasets/era5_weekly/1959-2023_01_10-full_37-1h-0p25deg-chunk-1-s2s.zarr/sea_surface_temperature/10004.0.0" \
  "gs://weatherbench2/datasets/era5_weekly/1959-2023_01_10-full_37-1h-0p25deg-chunk-1-s2s.zarr/sea_surface_temperature/10005.0.0" \
  "gs://weatherbench2/datasets/era5_weekly/1959-2023_01_10-full_37-1h-0p25deg-chunk-1-s2s.zarr/sea_surface_temperature/10006.0.0" \
  "gs://weatherbench2/datasets/era5_weekly/1959-2023_01_10-full_37-1h-0p25deg-chunk-1-s2s.zarr/sea_surface_temperature/10007.0.0" \
  "gs://weatherbench2/datasets/era5_weekly/1959-2023_01_10-full_37-1h-0p25deg-chunk-1-s2s.zarr/sea_surface_temperature/10008.0.0" \
  "gs://weatherbench2/datasets/era5_weekly/1959-2023_01_10-full_37-1h-0p25deg-chunk-1-s2s.zarr/sea_surface_temperature/10009.0.0" \
  "gs://weatherbench2/datasets/era5_weekly/1959-2023_01_10-full_37-1h-0p25deg-chunk-1-s2s.zarr/sea_surface_temperature/1001.0.0" \
  "gs://weatherbench2/datasets/era5_weekly/1959-2023_01_10-full_37-1h-0p25deg-chunk-1-s2s.zarr/sea_surface_temperature/10010.0.0" \
  "gs://weatherbench2/datasets/era5_weekly/1959-2023_01_10-full_37-1h-0p25deg-chunk-1-s2s.zarr/sea_surface_temperature/10011.0.0" \
  "gs://weatherbench2/datasets/era5_weekly/1959-2023_01_10-full_37-1h-0p25deg-chunk-1-s2s.zarr/sea_surface_temperature/10012.0.0" \
  "gs://weatherbench2/datasets/era5_weekly/1959-2023_01_10-full_37-1h-0p25deg-chunk-1-s2s.zarr/sea_surface_temperature/10013.0.0" \
  "gs://weatherbench2/datasets/era5_weekly/1959-2023_01_10-full_37-1h-0p25deg-chunk-1-s2s.zarr/sea_surface_temperature/10014.0.0" \
  "gs://weatherbench2/datasets/era5_weekly/1959-2023_01_10-full_37-1h-0p25deg-chunk-1-s2s.zarr/sea_surface_temperature/10015.0.0" \
  "gs://weatherbench2/datasets/era5_weekly/1959-2023_01_10-full_37-1h-0p25deg-chunk-1-s2s.zarr/sea_surface_temperature/10016.0.0" \
  "gs://weatherbench2/datasets/era5_weekly/1959-2023_01_10-full_37-1h-0p25deg-chunk-1-s2s.zarr/sea_surface_temperature/10017.0.0" \
  "gs://weatherbench2/datasets/era5_weekly/1959-2023_01_10-full_37-1h-0p25deg-chunk-1-s2s.zarr/sea_surface_temperature/10018.0.0" \
  "gs://weatherbench2/datasets/era5_weekly/1959-2023_01_10-full_37-1h-0p25deg-chunk-1-s2s.zarr/sea_surface_temperature/10019.0.0" \
  "gs://weatherbench2/datasets/era5_weekly/1959-2023_01_10-full_37-1h-0p25deg-chunk-1-s2s.zarr/sea_surface_temperature/1002.0.0" \
  "gs://weatherbench2/datasets/era5_weekly/1959-2023_01_10-full_37-1h-0p25deg-chunk-1-s2s.zarr/sea_surface_temperature/10020.0.0" \
  "gs://weatherbench2/datasets/era5_weekly/1959-2023_01_10-full_37-1h-0p25deg-chunk-1-s2s.zarr/sea_surface_temperature/10021.0.0" \
  "gs://weatherbench2/datasets/era5_weekly/1959-2023_01_10-full_37-1h-0p25deg-chunk-1-s2s.zarr/sea_surface_temperature/10022.0.0" \
  "gs://weatherbench2/datasets/era5_weekly/1959-2023_01_10-full_37-1h-0p25deg-chunk-1-s2s.zarr/sea_surface_temperature/10023.0.0" \
  "gs://weatherbench2/datasets/era5_weekly/1959-2023_01_10-full_37-1h-0p25deg-chunk-1-s2s.zarr/sea_surface_temperature/10024.0.0" \
  "gs://weatherbench2/datasets/era5_weekly/1959-2023_01_10-full_37-1h-0p25deg-chunk-1-s2s.zarr/sea_surface_temperature/10025.0.0" \
  "gs://weatherbench2/datasets/era5_weekly/1959-2023_01_10-full_37-1h-0p25deg-chunk-1-s2s.zarr/sea_surface_temperature/10026.0.0" \
  "gs://weatherbench2/datasets/era5_weekly/1959-2023_01_10-full_37-1h-0p25deg-chunk-1-s2s.zarr/sea_surface_temperature/10027.0.0" \
  "gs://weatherbench2/datasets/era5_weekly/1959-2023_01_10-full_37-1h-0p25deg-chunk-1-s2s.zarr/sea_surface_temperature/10028.0.0" \
  "gs://weatherbench2/datasets/era5_weekly/1959-2023_01_10-full_37-1h-0p25deg-chunk-1-s2s.zarr/sea_surface_temperature/10029.0.0" \
  "gs://weatherbench2/datasets/era5_weekly/1959-2023_01_10-full_37-1h-0p25deg-chunk-1-s2s.zarr/sea_surface_temperature/1003.0.0" \
  "gs://weatherbench2/datasets/era5_weekly/1959-2023_01_10-full_37-1h-0p25deg-chunk-1-s2s.zarr/sea_surface_temperature/10030.0.0" \
  "gs://weatherbench2/datasets/era5_weekly/1959-2023_01_10-full_37-1h-0p25deg-chunk-1-s2s.zarr/sea_surface_temperature/10031.0.0" \
  "gs://weatherbench2/datasets/era5_weekly/1959-2023_01_10-full_37-1h-0p25deg-chunk-1-s2s.zarr/sea_surface_temperature/10032.0.0" \
  "gs://weatherbench2/datasets/era5_weekly/1959-2023_01_10-full_37-1h-0p25deg-chunk-1-s2s.zarr/sea_surface_temperature/10033.0.0" \
  "gs://weatherbench2/datasets/era5_weekly/1959-2023_01_10-full_37-1h-0p25deg-chunk-1-s2s.zarr/sea_surface_temperature/10034.0.0" \
  "gs://weatherbench2/datasets/era5_weekly/1959-2023_01_10-full_37-1h-0p25deg-chunk-1-s2s.zarr/sea_surface_temperature/10035.0.0" \
  "gs://weatherbench2/datasets/era5_weekly/1959-2023_01_10-full_37-1h-0p25deg-chunk-1-s2s.zarr/sea_surface_temperature/10036.0.0" \
  "gs://weatherbench2/datasets/era5_weekly/1959-2023_01_10-full_37-1h-0p25deg-chunk-1-s2s.zarr/sea_surface_temperature/10037.0.0" \
  "gs://weatherbench2/datasets/era5_weekly/1959-2023_01_10-full_37-1h-0p25deg-chunk-1-s2s.zarr/sea_surface_temperature/10038.0.0" \
  "gs://weatherbench2/datasets/era5_weekly/1959-2023_01_10-full_37-1h-0p25deg-chunk-1-s2s.zarr/sea_surface_temperature/10039.0.0" \
  /mnt/qb/goswami/data/era5/weatherbench2/weekly_mean_sst