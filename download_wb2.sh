# gsutil -m cp -r "gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr" /mnt/qb/goswami/data/era5/weatherbench2/

# gsutil -m cp -r gs://weatherbench2/datasets/era5/1959-2023_01_10-1h-240x121_equiangular_with_poles_conservative.zarr gs://weatherbench2/datasets/era5-hourly-climatology/1990-2019_6h_1440x721.zarr gs://weatherbench2/datasets/era5-hourly-climatology/1990-2019_6h_240x121_equiangular_with_poles_conservative.zarr /mnt/qb/goswami/data/era5/weatherbench2/

# gsutil -m cp -r "gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr" /mnt/V/Master/data/weatherbench2/
# gsutil -m cp -r \
#   "gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr" \
#   .

gsutil -m cp -r \
  "gs://weatherbench2/datasets/era5/1959-2022-full_37-6h-0p25deg-chunk-1.zarr-v2" \
  .