# (C) Copyright 2023 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
from functools import cached_property
from datetime import datetime
import os
import climetlab as cml
import xarray as xr

LOG = logging.getLogger(__name__)

local_era5 = {
    "10u":"10m_u_component_of_wind/10m_u_component_of_wind_{}.nc",
    "10v":"10m_v_component_of_wind/10m_v_component_of_wind_{}.nc",
    "2t":"2m_dewpoint_temperature/2m_dewpoint_temperature_{}.nc",
    "sp":"surface_pressure/surface_pressure_{}.nc",
    "msl":"mean_sea_level_pressure/mean_sea_level_pressure_{}.nc",
    "tcwv":"total_column_water_vapour/total_column_water_vapour_{}.nc",
    "100u":"",
    "100v":"",
}

# local_era5_dict = {
#         "10u":"single_pressure_level/10m_u_component_of_wind_{}.nc",
#         "10v":"single_pressure_level/10m_v_component_of_wind_{}.nc",
#         "100u":"",
#         "100v":"",
#         "2t":"single_pressure_level/2m_temperature/2m_temperature_{}.nc",    
#         "sp":"single_pressure_level/surface_pressure",
#         "msl":"single_pressure_level/mean_sea_level_pressure",
#         "tcwv":"single_pressure_level/total_column_water_vapour",
#         "u50":"multi_pressure_level/u_component_of_wind",
#         "u100":"multi_pressure_level/u_component_of_wind",
#         "u150":"multi_pressure_level/u_component_of_wind",
#         "u200":"multi_pressure_level/u_component_of_wind",
#         "u250":"multi_pressure_level/u_component_of_wind",
#         "u300":"multi_pressure_level/u_component_of_wind",
#         "u400":"multi_pressure_level/u_component_of_wind",
#         "u500":"multi_pressure_level/u_component_of_wind",
#         "u600":"multi_pressure_level/u_component_of_wind",
#         "u700":"multi_pressure_level/u_component_of_wind",
#         "u850":"multi_pressure_level/u_component_of_wind",
#         "u925":"multi_pressure_level/u_component_of_wind",
#         "u1000":"multi_pressure_level/u_component_of_wind",
#         "v50":"multi_pressure_level/v_component_of_wind/50/v_component_of_wind_{}_50.nc",
#         "v100":"multi_pressure_level/v_component_of_wind/100/v_component_of_wind_{}_100.nc",
#         "v150":"multi_pressure_level/v_component_of_wind/150/v_component_of_wind_{}_150.nc",
#         "v200":"multi_pressure_level/v_component_of_wind/200/v_component_of_wind_{}_200.nc",
#         "v250":"multi_pressure_level/v_component_of_wind/250/v_component_of_wind_{}_250.nc",
#         "v300":"multi_pressure_level/v_component_of_wind/300/v_component_of_wind_{}_300.nc",
#         "v400":"multi_pressure_level/v_component_of_wind/400/v_component_of_wind_{}_400.nc",
#         "v500":"multi_pressure_level/v_component_of_wind/500/v_component_of_wind_{}_500.nc",
#         "v600":"multi_pressure_level/v_component_of_wind/600/v_component_of_wind_{}_600.nc",
#         "v700":"multi_pressure_level/v_component_of_wind/700/v_component_of_wind_{}_700.nc",
#         "v850":"multi_pressure_level/v_component_of_wind/850/v_component_of_wind_{}_850.nc",
#         "v925":"multi_pressure_level/v_component_of_wind/925/v_component_of_wind_{}_925.nc",
#         "v1000":"multi_pressure_level/v_component_of_wind/1000/v_component_of_wind_{}_1000.nc",
#         "z50":"multi_pressure_level/geopotential"
#         "z100":"multi_pressure_level/geopotential"
#         "z150":"multi_pressure_level/geopotential"
#         "z200":"multi_pressure_level/geopotential"
#         "z250":"multi_pressure_level/geopotential"
#         "z300":"multi_pressure_level/geopotential"
#         "z400":"multi_pressure_level/geopotential"
#         "z500":"multi_pressure_level/geopotential"
#         "z600":"multi_pressure_level/geopotential"
#         "z700":"multi_pressure_level/geopotential"
#         "z850":"multi_pressure_level/geopotential"
#         "z925":"multi_pressure_level/geopotential"
#         "z1000":"multi_pressure_level/geopotential"
#         "t50":"multi_pressure_level/temperature"
#         "t100":"multi_pressure_level/temperature"
#         "t150":"multi_pressure_level/temperature"
#         "t200":"multi_pressure_level/temperature"
#         "t250":"multi_pressure_level/temperature"
#         "t300":"multi_pressure_level/temperature"
#         "t400":"multi_pressure_level/temperature"
#         "t500":"multi_pressure_level/temperature"
#         "t600":"multi_pressure_level/temperature"
#         "t700":"multi_pressure_level/temperature"
#         "t850":"multi_pressure_level/temperature"
#         "t925":"multi_pressure_level/temperature"
#         "t1000":"multi_pressure_level/temperature"
#         "r50": "multi_pressure_level/specific_humidity"
#         "r100":"multi_pressure_level/specific_humidity"
#         "r150":"multi_pressure_level/specific_humidity"
#         "r200":"multi_pressure_level/specific_humidity"
#         "r250":"multi_pressure_level/specific_humidity"
#         "r300":"multi_pressure_level/specific_humidity"
#         "r400":"multi_pressure_level/specific_humidity"
#         "r500":"multi_pressure_level/specific_humidity"
#         "r600":"multi_pressure_level/specific_humidity"
#         "r700":"multi_pressure_level/specific_humidity"
#         "r850":"multi_pressure_level/specific_humidity"
#         "r925":"multi_pressure_level/specific_humidity"
#         "r1000":"multi_pressure_level/specific_humidity"
#     ]


class RequestBasedInput:
    def __init__(self, owner, **kwargs):
        self.owner = owner
        self.store_path = kwargs.get("input_store",None)

    def _patch(self, **kargs):
        r = dict(**kargs)
        self.owner.patch_retrieve_request(r)
        return r

    @cached_property
    def fields_sfc(self):
        LOG.info(f"Loading surface fields from {self.WHERE}")

        return cml.load_source(
            "multi",
            [
                self.sfc_load_source(
                    **self._patch(
                        date=date,
                        time=time,
                        param=self.owner.param_sfc,
                        grid=self.owner.grid,
                        area=self.owner.area,
                        **self.owner.retrieve,
                    )
                )
                for date, time in self.owner.datetimes()
            ],
        )

    @cached_property
    def fields_pl(self):
        LOG.info(f"Loading pressure fields from {self.WHERE}")
        param, level = self.owner.param_level_pl
        return cml.load_source(
            "multi",
            [
                self.pl_load_source(
                    **self._patch(
                        date=date,
                        time=time,
                        param=param,
                        level=level,
                        grid=self.owner.grid,
                        area=self.owner.area,
                    )
                )
                for date, time in self.owner.datetimes()
            ],
        )

    @cached_property
    def all_fields(self):
        if self.store_path is not None:
            LOG.info(f"Storing input data at {self.store_path}")
            self.fields_sfc.save(self.store_path)
            self.fields_pl.save(self.store_path)
        return self.fields_sfc + self.fields_pl


class MarsInput(RequestBasedInput):
    WHERE = "MARS"

    def __init__(self, owner, **kwargs):
        self.owner = owner

    def pl_load_source(self, **kwargs):
        kwargs["levtype"] = "pl"
        logging.debug("load source mars %s", kwargs)
        return cml.load_source("mars", kwargs)

    def sfc_load_source(self, **kwargs):
        kwargs["levtype"] = "sfc"
        logging.debug("load source mars %s", kwargs)
        return cml.load_source("mars", kwargs)


class CdsInput(RequestBasedInput):
    WHERE = "CDS"

    def pl_load_source(self, **kwargs):
        kwargs["product_type"] = "reanalysis"
        return cml.load_source("cds", "reanalysis-era5-pressure-levels", kwargs)

    def sfc_load_source(self, **kwargs):
        kwargs["product_type"] = "reanalysis"
        return cml.load_source("cds", "reanalysis-era5-single-levels", kwargs)

class LocalInput:
    def __init__(self,owner,era5_path, **kwargs):
        print("hi")
        self.era5_path = era5_path
        self.owner = owner

    def pl_load_source(self, **kwargs):
        return cml.load_source("file", os.path.join(self.path,"multi_pressure_level",local_era5[kwargs.param],local_era5[kwargs.level],local_era5[kwargs.param]+"_{}_"+local_era5[kwargs.level]).format(year))

    def sfc_load_source(self, **kwargs):
        return cml.load_source("file", os.path.join(self.path,"single_pressure_level",local_era5[kwargs.param]).format(year))

    def fields_sfc(self):
        LOG.info(f"Loading surface fields from {self.path}")
        parameters = []
        for date, time in self.owner.datetimes():
            date = datetime.strptime(str(date)+str(time).zfill(4),"%Y%m%d%H%M")
            for p in self.owner.param_sfc:
                parameters.append((year,month,day,time,p))
        return cml.load_source(
            "multi",
            [
                self.sfc_load_source(
                        date=date,
                        time=time,
                        param=param
                    )
                for date, time, param in parameters
            ],
        )

    def fields_pl(self):
        LOG.info(f"Loading pressure fields from {self.path}")
        param, level = self.owner.param_level_pl
        parameters = []
        for date, time in self.owner.datetimes():
            for p in param:
                for l in level:
                    parameters.append((date,time,p,l))
        return cml.load_source(
            "multi",
            [
                self.pl_load_source(
                        date=date,
                        time=time,
                        param=param,
                        level=level
                    )
                for date, time, param, level in parameters
            ],
        )

    def all_fields(self):
        return self.fields_sfc() + self.fields_pl()
    
    def all_fields(self):
        files = []
        for date, time in self.owner.datetimes():
            date = datetime.strptime(str(date)+str(time).zfill(4),"%Y%m%d%H%M")
            year = date.year
            for p in self.owner.param_sfc:
                files.append(os.path.join(self.era5_path,"single_pressure_level",local_era5[p],local_era5[p]+"_{}.nc").format(year))
            param, level = self.owner.param_level_pl
            for p in param:
                for l in level:
                    files.append(os.path.join(self.era5_path,"multi_pressure_level",local_era5[p],local_era5[l],local_era5[p]+"_{}_"+local_era5[l]+".nc").format(year))
        data = xr.open_mfdataset(files,parallel=True)
        for date, time in self.owner.datetimes():
            date = datetime.strptime(str(date)+str(time).zfill(4),"%Y%m%d%H%M")
            data_selected = data.sel(time=date)
        return data_selected

class FileInput:
    def __init__(self, owner, file, **kwargs):
        self.file = file
        self.owner = owner

    @cached_property
    def fields_sfc(self):
        return cml.load_source("file", self.file).sel(levtype="sfc")

    @cached_property
    def fields_pl(self):
        return cml.load_source("file", self.file).sel(levtype="pl")

    @cached_property
    def all_fields(self):
        return cml.load_source("file", self.file)


INPUTS = dict(
    mars=MarsInput,
    file=FileInput,
    cds=CdsInput,
    localERA5=LocalInput
)


def get_input(name, *args, **kwargs):
    return INPUTS[name](*args, **kwargs)


def available_inputs():
    return sorted(INPUTS.keys())
