# (C) Copyright 2023 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
from functools import cached_property

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
    def __init__(self,path):
        print("hi")
    def pl_load_source(self, **kwargs):
        kwargs["product_type"] = "reanalysis"
        return cml.load_source("cds", "reanalysis-era5-pressure-levels", kwargs)

    def sfc_load_source(self, **kwargs):
        kwargs["product_type"] = "reanalysis"
        return cml.load_source("file", "path/to/file"
            "cds", "reanalysis-era5-single-levels", kwargs)

def sfc_load_local(self, **kwargs):
    date = datetime.strptime(kwargs.date+str(kwargs.time),"%Y%m%d%H%M")
    # for p in kwargs.param:


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
    local=LocalInput
)


def get_input(name, *args, **kwargs):
    return INPUTS[name](*args, **kwargs)


def available_inputs():
    return sorted(INPUTS.keys())
