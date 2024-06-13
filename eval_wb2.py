import apache_beam   # Needs to be imported separately to avoid TypingError
import xarray as xr
import sys
import os
sys.path.append('/weatherbench2')
from weatherbench2 import config
from weatherbench2.metrics import MSE, ACC, Bias
from weatherbench2.evaluation import evaluate_in_memory, evaluate_with_beam

# wrong chunks
# file_name = 'forecast_lead_time=56_time=01.01.2016-01.01.2016.zarr'
# no shuffle
# file_name = 'forecast_lead_time=84_time=01.01.2016-01.01.2016.zarr'
#shuffle

# file_name = 'forecast_lead_time=84_time=2016-2018-shuffled.zarr'
# checkpoint_dir = '/mnt/qb/work2/goswami0/gkd965/checkpoints/solar-spaceship-111-sID{0005}/'
# obs_path = '/mnt/qb/goswami/data/era5/weatherbench2/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr'
# climatology_path = '/mnt/qb/goswami/data/era5/weatherbench2/1990-2019_6h_1440x721.zarr'


file_name = 'forecast_lead_time=112_time=2016-2018-shuffled.zarr'
checkpoint_dir = '/mnt/qb/work2/goswami0/gkd965/checkpoints/jolly-blaze-113-sID{13578}'
obs_path = '/mnt/qb/goswami/data/era5/weatherbench2/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr'
climatology_path = '/mnt/qb/goswami/data/era5/weatherbench2/1990-2019_6h_1440x721.zarr'


out_name = file_name.replace('.zarr', '_')

climatology = xr.open_zarr(climatology_path)


paths = config.Paths(
    forecast=os.path.join(checkpoint_dir,file_name),
    obs=obs_path,
    output_dir=os.path.join(checkpoint_dir,'eval'),   # Directory to save evaluation results
    output_file_prefix= out_name,
)

selection = config.Selection(
    variables=[
        'geopotential',
        '2m_temperature',
        'relative_humidity',
        'temperature',
        'v_component_of_wind',
        'total_column_water_vapour',
        'mean_sea_level_pressure',
        
    ],
    levels=[300, 850,925,1000],
    time_slice=slice('2016-01-01', '2017-12-31'),
)

data_config = config.Data(selection=selection, paths=paths)

eval_configs = {
    'eval' : config.Eval(
      metrics={
          'mse': MSE(), 
          'acc': ACC(climatology=climatology),
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