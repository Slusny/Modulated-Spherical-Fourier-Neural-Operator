
# pre relative humidty -------------------------------------------

python /home/goswami/gkd965/MasterML/main.py --save-forecast --model sfno --film-gen mae --model-version film --resume-checkpoint /mnt/qb/work2/goswami0/gkd965/checkpoints/solar-spaceship-111-sID{0005}/checkpoint_sfno_film_mae_iter=0_epoch=1.pkl --multi-step-validation 4 --validation-step-skip 27 --output-path /mnt/qb/work2/goswami0/gkd965/checkpoints/solar-spaceship-111-sID{0005} --no-shuffle  --num-iterations 73 --batch-size 5 --training-workers 12

# jolly-blaze-113 (1film)
python /home/goswami/gkd965/MasterML/main.py --save-forecast --model sfno --film-gen mae --model-version film --resume-checkpoint /mnt/qb/work2/goswami0/gkd965/checkpoints/jolly-blaze-113-sID{13578}/checkpoint_sfno_film_mae_iter=0_epoch=1.pkl --multi-step-validation 4 --validation-step-skip 27 --output-path /mnt/qb/work2/goswami0/gkd965/checkpoints/jolly-blaze-113-sID{13578} --num-iterations 73 --batch-size 5 --save-checkpoint-interval 20 --training-workers 12

# whole-microwave-125
python /home/goswami/gkd965/MasterML/main.py --save-forecast --model sfno --film-gen mae --model-version film --resume-checkpoint /mnt/qb/work2/goswami0/gkd965/checkpoints/whole-microwave-125-sID{32922}/checkpoint_sfno_film_mae_iter=0_epoch=1.pkl --multi-step-validation 4 --validation-step-skip 27 --output-path /mnt/qb/work2/goswami0/gkd965/checkpoints/whole-microwave-125-sID{32922} --num-iterations 73 --batch-size 5 --save-checkpoint-interval 20 --training-workers 12


# post relative humidty -------------------------------------------





# save era5
python main.py --model sfno --model-version latest  --save-data --trainingset-start-year 2018 --trainingset-end-year 2019 --no-shuffle --num_iterations 9 --output zarr --path /mnt/qb/goswami/data/era5 --batch-size 6






# Sfno (R - 214)
# python /home/goswami/gkd965/MasterML/main.py --save-forecast --model sfno --model-version latest --multi-step-validation 4 --validation-step-skip 27 --output-path /mnt/qb/work2/goswami0/gkd965/checkpoints/sfno/newdata --num-iterations 12 --batch-size 3 --save-checkpoint-interval 10 --training-workers 8 --validationset-start-year 2018 --validationset-end-year 2018 
python /home/goswami/gkd965/MasterML/main.py --save-forecast --model sfno --model-version latest --multi-step-validation 4 --validation-step-skip 27 --output-path /mnt/qb/goswami/data/era5/weatherbench2/res/sfno --num-iterations 1 --batch-size 3 --save-checkpoint-interval 24 --training-workers 0 --validationset-start-year 2018 --validationset-end-year 2019

# gcn
python /home/goswami/gkd965/MasterML/main.py --save-forecast --model sfno --model-version film --resume-checkpoint /mnt/qb/work2/goswami0/gkd965/checkpoints/lunar-terrain-5-sID{21191}/checkpoint_sfno_film_gcn_iter=798_epoch=0.pkl  --multi-step-validation 4 --validation-step-skip 27 --output-path /mnt/qb/work2/goswami0/gkd965/checkpoints/lunar-terrain-5-sID{21191} --num-iterations 12 --batch-size 3 --save-checkpoint-interval 10 --training-workers 6 --validationset-start-year 2018 --validationset-end-year 2018 --set-rank 1


#--------------- 1 Film
# rich-breeze (D) - (checkpoint)
python /home/goswami/gkd965/MasterML/main.py --save-forecast --model sfno --model-version film --resume-checkpoint /mnt/qb/work2/goswami0/gkd965/checkpoints/rich-breeze-23-sID{x01l12}/rich-breeze-23-sID{x01l13}/rich-breeze-23-sID{x01l14}/checkpoint_sfno_film_mae_iter=0_epoch=3.pkl  --multi-step-validation 4 --validation-step-skip 27 --output-path /mnt/qb/work2/goswami0/gkd965/checkpoints/rich-breeze-23-sID{x01l12}/rich-breeze-23-sID{x01l13}/rich-breeze-23-sID{x01l14} --num-iterations 12 --batch-size 3 --save-checkpoint-interval 10 --training-workers 6 --validationset-start-year 2018 --validationset-end-year 2018  #--set-rank 1
# Ones (R - 211)
python /home/goswami/gkd965/MasterML/main.py --save-forecast --model sfno --model-version film --resume-checkpoint /mnt/qb/work2/goswami0/gkd965/checkpoints/restful-valley-24-sID{x0_0_0}/restful-valley-24-ONES-sID{y0_0_0}/checkpoint_sfno_film_mae_iter=0_epoch=2.pkl  --multi-step-validation 4 --validation-step-skip 27 --output-path /mnt/qb/goswami/data/era5/weatherbench2/res/1film/ONES --num-iterations 1 --batch-size 3 --save-checkpoint-interval 24 --training-workers 0 --validationset-start-year 2018 --validationset-end-year 2019




#--------------- 2 Film

#restful-cherry (21Film) (R - 211)
python /home/goswami/gkd965/MasterML/main.py --save-forecast --model sfno --model-version film --resume-checkpoint /mnt/qb/work2/goswami0/gkd965/checkpoints/copper-fog-3-sID{x00112}/copper-fog-3-sID{x00115}/restful-cherry-7-sID{y00117}/checkpoint_sfno_film_mae_iter=0_epoch=1.pkl   --multi-step-validation 4 --validation-step-skip 27 --output-path /mnt/qb/goswami/data/era5/weatherbench2/res/11film/restful-cherry --num-iterations 1 --batch-size 3 --save-checkpoint-interval 24 --training-workers 0 --validationset-start-year 2018 --validationset-end-year 2019  #--set-rank 1


# fearless-pyramid (R - 211)
python /home/goswami/gkd965/MasterML/main.py --save-forecast --model sfno --model-version film --resume-checkpoint /mnt/qb/work2/goswami0/gkd965/checkpoints/fearless-pyramid-13-sID{x0112}/fearless-pyramid-13-sID{x0113}/fearless-pyramid-13-sID{x0114}/checkpoint_sfno_film_mae_iter=420_epoch=2.pkl   --multi-step-validation 4 --validation-step-skip 27 --output-path /mnt/qb/goswami/data/era5/weatherbench2/res/2film/fearless-pyramid --num-iterations 1 --batch-size 3 --save-checkpoint-interval 24 --training-workers 0 --validationset-start-year 2018 --validationset-end-year 2019  --set-rank 2




#--------------- vit (R - 214)
# python /home/goswami/gkd965/MasterML/main.py --save-forecast --model sfno --model-version film --resume-checkpoint /mnt/qb/work2/goswami0/gkd965/checkpoints/devoted-breeze-7-sID{080808}/devoted-breeze-7-sID{181818}/devoted-breeze-7-sID{282828}/checkpoint_sfno_film_transformer_iter=0_epoch=3.pkl  --multi-step-validation 4 --validation-step-skip 27 --output-path /mnt/qb/goswami/data/era5/weatherbench2/res/vit/devoted-breeze --num-iterations 1 --batch-size 3 --save-checkpoint-interval 24 --training-workers 0 --validationset-start-year 2018 --validationset-end-year 2019
python /home/goswami/gkd965/MasterML/main.py --save-forecast --model sfno --model-version film --resume-checkpoint /mnt/qb/work2/goswami0/gkd965/checkpoints/devoted-breeze-7-sID{080808}/devoted-breeze-7-sID{181818}/devoted-breeze-7-sID{282828}/checkpoint_sfno_film_transformer_iter=360_epoch=2.pkl  --multi-step-validation 4 --validation-step-skip 27 --output-path /mnt/qb/goswami/data/era5/weatherbench2/res/vit/devoted-breeze --num-iterations 1 --batch-size 3 --save-checkpoint-interval 24 --training-workers 0 --validationset-start-year 2018 --validationset-end-year 2019

#--------------- gcn (R - 211)
python /home/goswami/gkd965/MasterML/main.py --save-forecast --model sfno --model-version film --resume-checkpoint /mnt/qb/work2/goswami0/gkd965/checkpoints/lunar-terrain-5-sID{21191}/checkpoint_sfno_film_gcn_iter=798_epoch=0.pkl  --multi-step-validation 4 --validation-step-skip 27 --output-path /mnt/qb/goswami/data/era5/weatherbench2/res/gcn/lunar-terrain --num-iterations 1 --batch-size 3 --save-checkpoint-interval 24 --training-workers 0 --validationset-start-year 2018 --validationset-end-year 2019





# -- vanilla 

# damped

# greatfull field (R - 214)
python /home/goswami/gkd965/MasterML/main.py --save-forecast --model sfno --model-version film --resume-checkpoint /mnt/qb/work2/goswami0/gkd965/checkpoints/grateful-field-13-sID{859123}/checkpoint_sfno_film_mae_iter=0_epoch=1.pkl  --multi-step-validation 4 --validation-step-skip 27 --output-path /mnt/qb/goswami/data/era5/weatherbench2/res/vanilla/1film --num-iterations 1 --batch-size 3 --save-checkpoint-interval 24 --training-workers 0 --validationset-start-year 2018 --validationset-end-year 2019  #--set-rank 1

# proud-totem
python /home/goswami/gkd965/MasterML/main.py --save-forecast --model sfno --model-version film --resume-checkpoint /mnt/qb/work2/goswami0/gkd965/checkpoints/proud-totem-8-sID{715721}/checkpoint_sfno_film_mae_iter=0_epoch=1.pkl  --multi-step-validation 4 --validation-step-skip 27 --output-path /mnt/qb/goswami/data/era5/weatherbench2/res/vanilla/2film --num-iterations 1 --batch-size 3 --save-checkpoint-interval 24 --training-workers 0 --validationset-start-year 2018 --validationset-end-year 2019  #--set-rank 1

# visionary-sponge 12
python /home/goswami/gkd965/MasterML/main.py --save-forecast --model sfno --model-version film --resume-checkpoint /mnt/qb/work2/goswami0/gkd965/checkpoints/visionary-sponge-7-sID{2641}/checkpoint_sfno_film_mae_iter=0_epoch=1.pkl  --multi-step-validation 4 --validation-step-skip 27 --output-path /mnt/qb/goswami/data/era5/weatherbench2/res/vanilla/2film/visionary-sponge --num-iterations 1 --batch-size 3 --save-checkpoint-interval 24 --training-workers 0 --validationset-start-year 2018 --validationset-end-year 2019  #--set-rank 1

# -------------1film

# lunar-mountain (225)
python /home/goswami/gkd965/MasterML/main.py --save-forecast --model sfno --model-version film --resume-checkpoint /mnt/qb/work2/goswami0/gkd965/checkpoints/lunar-mountain-8-sID{659123}/checkpoint_sfno_film_mae_iter=0_epoch=1.pkl  --multi-step-validation 4 --validation-step-skip 27 --output-path /mnt/qb/goswami/data/era5/weatherbench2/res/vanilla/1film/lunar-montain --num-iterations 1 --batch-size 3 --save-checkpoint-interval 24 --training-workers 0 --validationset-start-year 2018 --validationset-end-year 2019  #--set-rank 1

# brisk brook (211)
python /home/goswami/gkd965/MasterML/main.py --save-forecast --model sfno --model-version film --resume-checkpoint /mnt/qb/work2/goswami0/gkd965/checkpoints/brisk-brook-7-sID{459123}/checkpoint_sfno_film_mae_iter=0_epoch=1.pkl  --multi-step-validation 4 --validation-step-skip 27 --output-path /mnt/qb/goswami/data/era5/weatherbench2/res/vanilla/1film/brisk-brook --num-iterations 1 --batch-size 3 --save-checkpoint-interval 24 --training-workers 0 --validationset-start-year 2018 --validationset-end-year 2019  #--set-rank 1

# fragrant grass (222)
python /home/goswami/gkd965/MasterML/main.py --save-forecast --model sfno --model-version film --resume-checkpoint /mnt/qb/work2/goswami0/gkd965/checkpoints/fragrant-grass-6-sID{31361}/checkpoint_sfno_film_mae_iter=0_epoch=1.pkl  --multi-step-validation 4 --validation-step-skip 27 --output-path /mnt/qb/goswami/data/era5/weatherbench2/res/vanilla/1film/fragrant-grass --num-iterations 1 --batch-size 3 --save-checkpoint-interval 24 --training-workers 0 --validationset-start-year 2018 --validationset-end-year 2019  #--set-rank 1

# crimson (210)
python /home/goswami/gkd965/MasterML/main.py --save-forecast --model sfno --model-version film --resume-checkpoint /mnt/qb/work2/goswami0/gkd965/checkpoints/crimson-yogurt-4-sID{54109}/checkpoint_sfno_film_mae_iter=0_epoch=1.pkl  --multi-step-validation 4 --validation-step-skip 27 --output-path /mnt/qb/goswami/data/era5/weatherbench2/res/vanilla/1film/crimson --num-iterations 1 --batch-size 3 --save-checkpoint-interval 24 --training-workers 0 --validationset-start-year 2018 --validationset-end-year 2019  #--set-rank 1


# -------------2film    

# atomic wind (222)
python /home/goswami/gkd965/MasterML/main.py --save-forecast --model sfno --model-version film --resume-checkpoint /mnt/qb/work2/goswami0/gkd965/checkpoints/atomic-wind-2-sID{615721}/checkpoint_sfno_film_mae_iter=0_epoch=1.pkl  --multi-step-validation 4 --validation-step-skip 27 --output-path /mnt/qb/goswami/data/era5/weatherbench2/res/vanilla/2film/atomic-wind --num-iterations 1 --batch-size 3 --save-checkpoint-interval 24 --training-workers 0 --validationset-start-year 2018 --validationset-end-year 2019  #--set-rank 1

# wandering lion (211)
python /home/goswami/gkd965/MasterML/main.py --save-forecast --model sfno --model-version film --resume-checkpoint /mnt/qb/work2/goswami0/gkd965/checkpoints/wandering-lion-1-sID{99988}/checkpoint_sfno_film_mae_iter=0_epoch=1.pkl  --multi-step-validation 4 --validation-step-skip 27 --output-path /mnt/qb/goswami/data/era5/weatherbench2/res/vanilla/1film/wandering-lion --num-iterations 1 --batch-size 3 --save-checkpoint-interval 24 --training-workers 0 --validationset-start-year 2018 --validationset-end-year 2019  #--set-rank 1

# dauntless rive (255)
python /home/goswami/gkd965/MasterML/main.py --save-forecast --model sfno --model-version film --resume-checkpoint /mnt/qb/work2/goswami0/gkd965/checkpoints/dauntless-river-3-sID{1641}/checkpoint_sfno_film_mae_iter=0_epoch=1.pkl  --multi-step-validation 4 --validation-step-skip 27 --output-path /mnt/qb/goswami/data/era5/weatherbench2/res/vanilla/1film/dauntless-river --num-iterations 1 --batch-size 3 --save-checkpoint-interval 24 --training-workers 0 --validationset-start-year 2018 --validationset-end-year 2019  #--set-rank 1




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