import apache_beam   # Needs to be imported separately to avoid TypingError
import xarray as xr
import sys
import os
sys.path.append('/weatherbench2')
from weatherbench2 import config
from weatherbench2.metrics import MSE, RMSE, ACC, Bias, SpatialMAE
from weatherbench2.evaluation import evaluate_in_memory, evaluate_with_beam
import argparse


parser = argparse.ArgumentParser()
# parser.add_argument(
#     "--obs-path",
#     action="store",
#     default='/mnt/qb/goswami/data/era5/weatherbench2/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr'
# )
parser.add_argument(
    "--checkpoint-dir",
    action="store",
    default='/mnt/qb/goswami/data/era5/weatherbench2/res/vanilla/1film/crimson/'
)
parser.add_argument(
    "--file-name",
    action="store",
    default='forecast_checkpoint_sfno_film_mae_iter=0_epoch=1_lead_time=112_time=2018-2019_shuffled_denormalised.zarr'
)
args = parser.parse_args()
file_name = args.file_name
checkpoint_dir = args.checkpoint_dir
obs_path = args.obs_path

#! ----------------
# need test dataset 2018
#! ----------------

# wrong chunks
# file_name = 'forecast_lead_time=56_time=01.01.2016-01.01.2016.zarr'
# no shuffle
# file_name = 'forecast_lead_time=84_time=01.01.2016-01.01.2016.zarr'
#shuffle

# file_name = 'forecast_lead_time=84_time=2016-2018-shuffled.zarr'
# checkpoint_dir = '/mnt/qb/work2/goswami0/gkd965/checkpoints/solar-spaceship-111-sID{0005}/'
# obs_path = '/mnt/qb/goswami/data/era5/weatherbench2/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr'
# climatology_path = '/mnt/qb/goswami/data/era5/weatherbench2/1990-2019_6h_1440x721.zarr'

#ERA5
# obs_path = '/mnt/qb/goswami/data/era5/weatherbench2/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr'
# normalized
obs_path =  '/mnt/qb/goswami/data/era5/era5_data_normalised_sfno_01.01.2016_31.12.2017.zarr'

climatology_path = '/mnt/qb/goswami/data/era5/weatherbench2/1990-2019_6h_1440x721.zarr'

# # jolly-blaze
# file_name = 'forecast_lead_time=112_steps=300_time=2016-2018-shuffled_denormalised.zarr'
# checkpoint_dir = '/mnt/qb/work2/goswami0/gkd965/checkpoints/jolly-blaze-113-sID{13578}'

# #snfo
# file_name = 'forecast_lead_time=112_steps=350_time=2016-2018-shuffled_denormalised.zarr'
# checkpoint_dir = '/mnt/qb/work2/goswami0/gkd965/checkpoints/sfno'

#snfo2
# file_name = 'forecast_weights_lead_time=112_time=2018-2019_shuffled_denormalised.zarr'
# checkpoint_dir = '/mnt/qb/goswami/data/era5/weatherbench2/res/sfno'


# microwave
# file_name = 'forecastcheckpoint_sfno_film_mae_iter=0_epoch=1_lead_time=112_steps=200_time=2016-2018-shuffled_denormalised.zarr'
# checkpoint_dir = '/mnt/qb/work2/goswami0/gkd965/checkpoints/whole-microwave-125-sID{32922}/'


# # rich-breeze
# file_name = 'forecast_checkpoint_sfno_film_mae_iter=0_epoch=3_lead_time=112_time=2018-2019_shuffled_denormalised.zarr'
# checkpoint_dir = '/mnt/qb/work2/goswami0/gkd965/checkpoints/rich-breeze-23-sID{x01l12}/rich-breeze-23-sID{x01l13}/rich-breeze-23-sID{x01l14}/'

# ONES
# file_name = 'forecast_checkpoint_sfno_film_mae_iter=0_epoch=2_lead_time=112_time=2018-2019_shuffled_denormalised.zarr'
# checkpoint_dir = '/mnt/qb/goswami/data/era5/weatherbench2/res/1film/ONES/'

# # lunar-terrain
# file_name = 'forecast_checkpoint_sfno_film_gcn_iter=798_epoch=0_lead_time=112_time=2018-2019_shuffled_denormalised.zarr'
# checkpoint_dir = '/mnt/qb/goswami/data/era5/weatherbench2/res/gcn/lunar-terrain'


# # fearless_pyramid
# file_name = 'forecast_checkpoint_sfno_film_mae_iter=420_epoch=2_lead_time=112_time=2018-2019_shuffled_denormalised.zarr'
# checkpoint_dir = '/mnt/qb/goswami/data/era5/weatherbench2/res/2film/fearless-pyramid/'

# # restful_cherry (R)
# file_name = 'forecast_checkpoint_sfno_film_mae_iter=0_epoch=1_lead_time=112_time=2018-2019_shuffled_denormalised.zarr'
# checkpoint_dir = '/mnt/qb/goswami/data/era5/weatherbench2/res/21film/restful-cherry/'

# # devoted_breeze
# file_name = 'forecast_checkpoint_sfno_film_transformer_iter=360_epoch=2_lead_time=112_time=2018-2019_shuffled_denormalised.zarr'
# checkpoint_dir = '/mnt/qb/goswami/data/era5/weatherbench2/res/vit/devoted-breeze/'

# # grateful_field
# file_name = 'forecast_checkpoint_sfno_film_mae_iter=0_epoch=1_lead_time=112_time=2018-2019_shuffled_denormalised.zarr'
# checkpoint_dir = '/mnt/qb/goswami/data/era5/weatherbench2/res/vanilla/1film/grateful-field/'

# # proud_totem
# file_name = 'forecast_checkpoint_sfno_film_mae_iter=0_epoch=1_lead_time=112_time=2018-2019_shuffled_denormalised.zarr'
# checkpoint_dir = '/mnt/qb/goswami/data/era5/weatherbench2/res/vanilla/2film/proud-totem/'

# # visionary_sponge
# file_name = 'forecast_checkpoint_sfno_film_mae_iter=0_epoch=1_lead_time=112_time=2018-2019_shuffled_denormalised.zarr'
# checkpoint_dir = '/mnt/qb/goswami/data/era5/weatherbench2/res/vanilla/2film/visionary-sponge/'

# # lunar_mountain
# file_name = 'forecast_checkpoint_sfno_film_mae_iter=0_epoch=1_lead_time=112_time=2018-2019_shuffled_denormalised.zarr'
# checkpoint_dir = '/mnt/qb/goswami/data/era5/weatherbench2/res/vanilla/1film/lunar-montain/'

# # brisk_brook
# file_name = 'forecast_checkpoint_sfno_film_mae_iter=0_epoch=1_lead_time=112_time=2018-2019_shuffled_denormalised.zarr'
# checkpoint_dir = '/mnt/qb/goswami/data/era5/weatherbench2/res/vanilla/1film/brisk-brook/'

# # fragrant_grass
# file_name = 'forecast_checkpoint_sfno_film_mae_iter=0_epoch=1_lead_time=112_time=2018-2019_shuffled_denormalised.zarr'
# checkpoint_dir = '/mnt/qb/goswami/data/era5/weatherbench2/res/vanilla/1film/fragrant-grass/'

## crimson
# file_name = 'forecast_checkpoint_sfno_film_mae_iter=0_epoch=1_lead_time=112_time=2018-2019_shuffled_denormalised.zarr'
# checkpoint_dir = '/mnt/qb/goswami/data/era5/weatherbench2/res/vanilla/1film/crimson/'



## atomic_wind
# file_name = 'forecast_checkpoint_sfno_film_mae_iter=0_epoch=1_lead_time=112_time=2018-2019_shuffled_denormalised.zarr'
# checkpoint_dir = '/mnt/qb/goswami/data/era5/weatherbench2/res/vanilla/2film/atomic-wind/'

## wandering_lion
# file_name = 'forecast_checkpoint_sfno_film_mae_iter=0_epoch=1_lead_time=112_time=2018-2019_shuffled_denormalised.zarr'
# checkpoint_dir = '/mnt/qb/goswami/data/era5/weatherbench2/res/vanilla/1film/wandering-lion/'

## dauntless_rive
# file_name = 'forecast_checkpoint_sfno_film_mae_iter=0_epoch=1_lead_time=112_time=2018-2019_shuffled_denormalised.zarr'
# checkpoint_dir = '/mnt/qb/goswami/data/era5/weatherbench2/res/vanilla/1film/dauntless-river/'

# 
# 
out_name = file_name.replace('.zarr', '_')

climatology = xr.open_zarr(climatology_path)


paths = config.Paths(
    forecast=os.path.join(checkpoint_dir,file_name),
    obs=obs_path,
    output_dir=os.path.join(checkpoint_dir,'eval_onlyMSE'),   # Directory to save evaluation results
    output_file_prefix= out_name,
)

# ERA5
selection = config.Selection(
    variables=[
        'geopotential',
        'relative_humidity',
        'temperature',
        'v_component_of_wind',
        'u_component_of_wind',
        "10m_u_component_of_wind", 
        "10m_v_component_of_wind",  
        "100m_u_component_of_wind",
        "100m_v_component_of_wind",
        "surface_pressure", 
        '2m_temperature',
        'total_column_water_vapour',
        'mean_sea_level_pressure',
    ],
    levels=[1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50],
    time_slice=slice('2018-01-01', '2018-12-31'),
)
# # RH
# selection = config.Selection(
#     variables=[
#         'relative_humidity',
#     ],
#     levels=[1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50],
#     time_slice=slice('2018-01-01', '2018-12-31'),
# )
# #U100
# selection = config.Selection(
#     variables=[
#         'u100',
#         'v100',
#     ],
#     time_slice=slice('2018-01-01', '2018-12-31'),
# )

data_config = config.Data(selection=selection, paths=paths)

# data_config = config.Data( paths=paths)

eval_configs = {
    'eval' : config.Eval(
      metrics={
          'mse': RMSE(), 
        #   'special_mse': SpatialMAE(),
        #   'acc': ACC(climatology=climatology),
          'bias': Bias(),
      },
      output_format='zarr',
  )
}

# evaluate_with_beam(
#     data_config,
#     eval_configs,
#     runner='DirectRunner',
#     num_threads=5,
#     input_chunks={'time': 1},
# )

evaluate_in_memory(data_config, eval_configs)

print("done evaluating")