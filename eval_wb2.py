import apache_beam   # Needs to be imported separately to avoid TypingError
import xarray as xr
import sys
sys.path.append('/weatherbench2')
from weatherbench2 import config
from weatherbench2.metrics import MSE, ACC, Bias
from weatherbench2.evaluation import evaluate_in_memory, evaluate_with_beam

forecast_path = '/mnt/qb/work2/goswami0/gkd965/checkpoints/solar-spaceship-111-sID{0005}/forecast_lead_time=84_time=2016-2018-shuffled.zarr'
obs_path = '/mnt/qb/goswami/data/era5/weatherbench2/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr'
climatology_path = '/mnt/qb/goswami/data/era5/weatherbench2/1990-2019_6h_1440x721.zarr'

climatology = xr.open_zarr(climatology_path)

paths = config.Paths(
    forecast=forecast_path,
    obs=obs_path,
    output_dir='/mnt/qb/work2/goswami0/gkd965/checkpoints/solar-spaceship-111-sID{0005}/eval',   # Directory to save evaluation results
)

selection = config.Selection(
    variables=[
        'geopotential',
        '2m_temperature'
    ],
    levels=[500, 700, 850],
    time_slice=slice('2016-01-01', '2017-12-31'),
)

data_config = config.Data(selection=selection, paths=paths)


eval_configs = {
  'deterministic': config.Eval(
      metrics={
          'mse': MSE(), 
          'acc': ACC(climatology=climatology),
          'bias': Bias(),
      },
  )
}

evaluate_with_beam(
    data_config,
    eval_configs,
    runner='DirectRunner',
    num_threads=5,
    input_chunks={'time': 20},
)