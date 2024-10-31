# This script copies the relevant varibles from a downloaded weatherbench2_era5 dataset to the a Target Directory.
# You can additionally prune the dataset in the time dimension to further reduce the used disk space.

DATASET_DIR='/mnt/qb/goswami/data/era5/weatherbench2/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr'
TARGET_DIR="/mnt/qb/work2/goswami0/gkd965/inputs/wb2_2001-2003.zarr"


DATASET_u100='/mnt/qb/goswami/data/era5/u100m_v100m_721x1440/u100m_1959-2022_721x1440_correct_chunk_new_mean_INTERPOLATE.zarr'
DATASET_v100='/mnt/qb/goswami/data/era5/u100m_v100m_721x1440/v100m_1959-2023-10_721x1440_correct_chunk_new_mean_INTERPOLATE.zarr'
TARGET_DIR_u100="/mnt/qb/work2/goswami0/gkd965/inputs/u100_2001-2003.zarr"
TARGET_DIR_v100="/mnt/qb/work2/goswami0/gkd965/inputs/v100_2001-2003.zarr"

COORDS=(
    "time"
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
    "total_column_water_vapour"
)

VARIABLESPL=(
    "relative_humidity"
    "geopotential"
    "temperature"
    "u_component_of_wind"
    "v_component_of_wind"
)

set -x

# specify the start and end indices for the time slice. E.g. :
# start_time=59900 # 2000-01-01 00:00:00 - 2000 is a leap year
# end_time=62823 # 2001-12-31 18:00:00
start_time=61364 # 2001-01-01 00:00:00 
end_time=64283 # 2002-12-31 18:00:00

TIMES=($(seq $start_time 1 $end_time))

PL_TIMES=( "${TIMES[@]/%/.0.0.0}" )
SCF_TIMES=( "${TIMES[@]/%/.0.0}" )

# Copy WB2
copy_wb(){
    mkdir -p $TARGET_DIR
    cd $TARGET_DIR
    mkdir -p "${VARIABLESSCF[@]}" "${VARIABLESPL[@]}" "${COORDS[@]}"
    cp $DATASET_DIR/.z* ./
    gsutil -q -m cp -r  "${COORDS[@]/#/$DATASET_DIR/}" ./

    for var in "${VARIABLESSCF[@]}"; do
        VARIABLES_SCFT=()
        for time in "${SCF_TIMES[@]}"; do
            VARIABLES_SCFT+=("$var/$time")
        done
        gsutil -q -m cp -r  "${VARIABLES_SCFT[@]/#/$DATASET_DIR/}" $TARGET_DIR/$var/
    done

    for var in "${VARIABLESPL[@]}"; do
        VARIABLES_PLT=()
        for time in "${PL_TIMES[@]}"; do
            VARIABLES_PLT+=("$var/$time")
        done
        gsutil -q -m cp -r  "${VARIABLES_PLT[@]/#/$DATASET_DIR/}" $TARGET_DIR/$var/
    done
}

# Copy u100m and v100m
copy_wind(){ #$1 DATASET_DIR $2 TARGET_DIR $3 var
    mkdir -p $2
    cd $2
    mkdir -p "$3" "${COORDS[@]}"
    cp $1/.z* ./
    gsutil -q -m cp -r  "${COORDS[@]/#/$1/}" ./


    VARIABLES_SCFT=()
    for time in "${SCF_TIMES[@]}"; do
        VARIABLES_SCFT+=("$3/$time")
    done
    gsutil -q -m cp -r  "${VARIABLES_SCFT[@]/#/$1/}" $2/$3/

}

copy_wind $DATASET_u100 $TARGET_DIR_u100 u100
copy_wind $DATASET_v100 $TARGET_DIR_v100 v100