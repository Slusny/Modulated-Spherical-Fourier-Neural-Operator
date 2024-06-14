python /home/goswami/gkd965/MasterML/main.py --save-forecast --model sfno --film-gen mae --model-version film --resume-checkpoint /mnt/qb/work2/goswami0/gkd965/checkpoints/solar-spaceship-111-sID{0005}/checkpoint_sfno_film_mae_iter=0_epoch=1.pkl --multi-step-validation 4 --validation-step-skip 27 --output-path /mnt/qb/work2/goswami0/gkd965/checkpoints/solar-spaceship-111-sID{0005} --no-shuffle  --num-iterations 73 --batch-size 5 --training-workers 12

# jolly-blaze-113 (1film)
python /home/goswami/gkd965/MasterML/main.py --save-forecast --model sfno --film-gen mae --model-version film --resume-checkpoint /mnt/qb/work2/goswami0/gkd965/checkpoints/jolly-blaze-113-sID{13578}/checkpoint_sfno_film_mae_iter=0_epoch=1.pkl --multi-step-validation 4 --validation-step-skip 27 --output-path /mnt/qb/work2/goswami0/gkd965/checkpoints/jolly-blaze-113-sID{13578} --num-iterations 73 --batch-size 5 --save-checkpoint-interval 20 --training-workers 12

# whole-microwave-125
python /home/goswami/gkd965/MasterML/main.py --save-forecast --model sfno --film-gen mae --model-version film --resume-checkpoint /mnt/qb/work2/goswami0/gkd965/checkpoints/whole-microwave-125-sID{32922}/checkpoint_sfno_film_mae_iter=0_epoch=1.pkl --multi-step-validation 4 --validation-step-skip 27 --output-path /mnt/qb/work2/goswami0/gkd965/checkpoints/whole-microwave-125-sID{32922} --num-iterations 73 --batch-size 5 --save-checkpoint-interval 20 --training-workers 12



# Sfno
python /home/goswami/gkd965/MasterML/main.py --save-forecast --model sfno --model-version latest --multi-step-validation 4 --validation-step-skip 27 --output-path /mnt/qb/work2/goswami0/gkd965/checkpoints/sfno  --num-iterations 73 --batch-size 5 --save-checkpoint-interval 10 --training-workers 8

#73*5=365
# 36*5=180

# INFO
# Multi step validation is 4 , skip 27 -> 4 wochen, 28 Tage, prediction_timedelta:5
# time= 180 -> 265GB

# - checkpoints/sfno/forecast_lead_time=112_time=2016-2018-shuffled.zarr
# - checkpoints/jolly-blaze-113-sID{13578}/forecast_lead_time=112_time=2016-2018-shuffled.zarr
# - 

# ERA5
# - /mnt/qb/goswami/data/era5/era5_data_normalised_sfno_01.01.2016_31.12.2017.zarr 861GB