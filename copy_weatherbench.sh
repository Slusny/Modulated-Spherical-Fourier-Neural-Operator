#!/bin/bash

# This copies the weatherbench2 data to the $SCRATCH directory

DATASET_DIR='/mnt/qb/goswami/data/era5/weatherbench2/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr'
TARGET_DIR="/mnt/qb/work2/goswami0/gkd965/inputs/wb2.zarr"

CLIMATOLOGY_DIR='/mnt/qb/goswami/data/WeatherBench2/1990-2017-daily_clim_daily_mean_61_dw_240x121_equiangular_with_poles_conservative.zarr'
CLIM_TARGET_DIR="$SCRATCH/wb2_clim.zarr"

COORDS=(
    "time"
    "level"
    "latitude"
    "longitude"
)

CLIM_COORDS=(
    "dayofyear"
    "level"
    "latitude"
    "longitude"
)

VARIABLESSCF=(
    "2m_temperature"
    "surface_pressure"
    "10m_u_component_of_wind"
    "10m_v_component_of_wind"
    "mean_sea_level_pressure"
    "sea_surface_temperature"
    "sea_ice_cover"
    "total_cloud_cover"
    "total_precipitation_6hr"
    "total_column_water_vapour"
)

VARIABLESPL=(
    "relative_humidity"
    "geopotential"
    "temperature"
    "u_component_of_wind"
    "v_component_of_wind"
)

STD_VARIABLES=("${VARIABLES[@]/%/_std}")

copy_zarr() {
    set -x
    mkdir -p $2
    cd $2

    # Copy metadata
    cp $1/.z* ./

    # Create directories for each variable (otherwise gsutil fails)
    mkdir "${VARIABLES[@]}" "${COORDS[@]}"

    # Copy each coordinate
    gsutil -m cp -r  "${COORDS[@]/#/$1/}" ./

    # Copy each variable
    gsutil -m cp -r "${VARIABLES[@]/#/$1/}" ./
    if [ "$3" = true ]; then
        mkdir "${VARIABLES[@]/%/_std}"
        gsutil -m cp -r "${VARIABLES[@]/%/_std}" ./
    fi
}

set -x

start_time=59947
# end_time=61408
end_time=59949
TIMES=($(seq $start_time 1 $end_time))

PL_TIMES=( "${TIMES[@]/%/.0.0.0}" )
SCF_TIMES=( "${TIMES[@]/%/.0.0}" )


# for each time in PL_TIMES append the time to each variable in VARIABLESPL
VARIABLES_PLT=()
for time in "${PL_TIMES[@]}"; do
    for var in "${VARIABLESPL[@]}"; do
        VARIABLES_PLT+=("$var/$time")
    done
done

# VARIABLES_SCFT=()
# for time in "${SCF_TIMES[@]}"; do
#     for var in "${VARIABLESSCF[@]}"; do
#         VARIABLES_SCFT+=("$var/$time")
#     done
# done


# 59947 - 61408 - 62869

# # Copy climatology
# mkdir -p $CLIM_TARGET_DIR
# cd $CLIM_TARGET_DIR
# mkdir -p "${VARIABLES[@]}" "${STD_VARIABLES[@]}" "${CLIM_COORDS[@]}"
# cp $CLIMATOLOGY_DIR/.z* ./
# gsutil -q -m cp -r  "${CLIM_COORDS[@]/#/$CLIMATOLOGY_DIR/}" "${VARIABLES[@]/#/$CLIMATOLOGY_DIR/}" "${STD_VARIABLES[@]/#/$CLIMATOLOGY_DIR/}" ./

# Copy WB2
mkdir -p $TARGET_DIR
cd $TARGET_DIR
mkdir -p "${VARIABLESSCF[@]}" "${VARIABLESPL[@]}" "${COORDS[@]}"
cp $DATASET_DIR/.z* ./
# gsutil -q -m cp -r  "${COORDS[@]/#/$DATASET_DIR/}" "${VARIABLES[@]/#/$DATASET_DIR/}" ./

# gsutil -q -m cp -r  "${COORDS[@]/#/$DATASET_DIR/}" "${VARIABLES_PLT[@]/#/$DATASET_DIR/}" "${VARIABLES_SCFT[@]/#/$DATASET_DIR/}" ./
# gsutil -q -m cp -r  "${VARIABLES_PLT[@]/#/$DATASET_DIR/}" "${VARIABLES_PLT[@]/#/$TARGET_DIR/}" 

for var in "${VARIABLESSCF[@]}"; do
    VARIABLES_SCFT=()
    for time in "${SCF_TIMES[@]}"; do
        VARIABLES_SCFT+=("$var/$time")
    done
    gsutil -q -m cp -r  "${VARIABLES_SCFT[@]/#/$DATASET_DIR/}" "$TARGET_DIR/$var/}" 
done
