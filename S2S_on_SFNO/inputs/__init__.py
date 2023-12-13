# (C) Copyright 2023 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
from functools import cached_property
import datetime
import os
import climetlab as cml

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


class RequestBasedInput:
    def __init__(self, owner, **kwargs):
        self.owner = owner

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
        self.path = era5_path
        self.owner = owner

    def pl_load_source(self, **kwargs):
        return cml.load_source("file", os.path.join(self.path,"multi_pressure_level",local_era5[kwargs.param],local_era5[kwargs.level],local_era5[kwargs.param]+"_{}_"+local_era5[kwargs.level]).format(year))

    def sfc_load_source(self, **kwargs):
        return cml.load_source("file", os.path.join(self.path,"single_pressure_level",local_era5[kwargs.param]).format(year))

    def fields_sfc(self):
        LOG.info(f"Loading surface fields from {self.WHERE}")
        parameters = []
        for date, time in self.owner.datetimes():
            date = datetime.strptime(date+str(time),"%Y%m%d%H%M")
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
        LOG.info(f"Loading pressure fields from {self.WHERE}")
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
