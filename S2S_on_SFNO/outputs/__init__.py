# (C) Copyright 2023 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
import xarray as xr
import climetlab as cml
import os
import numpy as np
LOG = logging.getLogger(__name__)


class FileOutput:
    def __init__(self, owner, path, metadata, **kwargs):
        self._first = True
        metadata.setdefault("expver", owner.expver)
        metadata.setdefault("class", "ml")

        LOG.info("Writing results to %s.", path)
        self.path = path
        self.owner = owner

        edition = metadata.pop("edition", 2)

        self.output = cml.new_grib_output(
            path,
            split_output=True,
            edition=edition,
            generatingProcessIdentifier=self.owner.version,
            **metadata,
        )

    def write(self, output,check_nans, template, step,**kwargs):
        for k, fs in enumerate(template):
            self.output.write(
                output[k, ...], check_nans=check_nans, template=fs, step=step
            )
    
class NetCDFOutput:
    # create new folder and save each step in a seperate netcdf file
    # combine them later during evaluation etc with xarray.open_mfdataset(./*)
    def __init__(self, owner, path, metadata, **kwargs):
        self._first = True
        metadata.setdefault("expver", owner.expver)
        metadata.setdefault("class", "ml")

        LOG.info("Writing results to %s.", path)
        self.path = path
        self.owner = owner

        edition = metadata.pop("edition", 2)

        pathList = path.split('/')
        pathDir = '/'.join(pathList[:-1])
        self.pathString = pathList[-1].split('.')[0]
        self.subdir = os.path.join(pathDir,self.pathString)
        os.makedirs(self.subdir, exist_ok=True)

    def write(self, output,check_nans, template,step,param_level_pl,param_sfc):
        # copy input data (template) once to copy metadata into output dataset
        if self.dataset is None:
            self.dataset = xr.zeros_like(template.to_xarray())
        
        # loop through variables and preassure levels and overwrite dataset with output from  model
        k = 0
        for sfc in param_sfc:
            axistupel = tuple(range(len(self.dataset[sfc].shape) - 2))
            self.dataset[sfc].values = np.expand_dims(output[k],axis=axistupel)
            k += 1
        pls, levels = param_level_pl
        levels.reverse()
        for pl in pls:
            for level in levels:
                axistupel = tuple(range(len(self.dataset[pl].shape) - 3))
                self.dataset[pl].sel(isobaricInhPa=level).values = np.expand_dims(output[k],axis=axistupel)
                k += 1

        self.dataset = self.dataset.assign_coords(step=[np.timedelta64(step*60*60*10**9, 'ns')])
        return self.dataset.to_netcdf(os.path.join(self.subdir, self.pathString + '_step_'+str(step)+'.nc'))



class HindcastReLabel:
    def __init__(self, owner, output, hindcast_reference_year, **kwargs):
        self.owner = owner
        self.output = output
        self.hindcast_reference_year = int(hindcast_reference_year)

    def write(self, output, template, step):
        for k, fs in enumerate(template):
            self.array_write(
                output[k, ...], check_nans=True, template=fs, step=step
            )

    def array_write(self, *args, **kwargs):
        if "date" in kwargs:
            date = kwargs["date"]
        else:
            date = kwargs["template"]["date"]

        assert len(str(date)) == 8
        date = int(date)
        referenceDate = self.hindcast_reference_year * 10000 + date % 10000

        kwargs.pop("date", None)
        kwargs["referenceDate"] = referenceDate
        kwargs["hdate"] = date
        return self.output.write(*args, **kwargs)


class NoneOutput:
    def __init__(self, *args, **kwargs):
        LOG.info("Results will not be written.")

    def write(self, *args, **kwargs):
        pass


OUTPUTS = dict(
    grib=FileOutput,
    netcdf=NetCDFOutput,
    none=NoneOutput,
)


def get_output(name, owner, *args, **kwargs):
    result = OUTPUTS[name](owner, *args, **kwargs)
    if kwargs.get("hindcast_reference_year") is not None:
        result = HindcastReLabel(owner, result, **kwargs)
    return result


def available_outputs():
    return sorted(OUTPUTS.keys())