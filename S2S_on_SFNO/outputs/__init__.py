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
import json
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


        try:
            with open(kwargs["output_variables"]) as f:
                self.output_variables = json.load(f)
            
            if type(self.output_variables) != list:
                raise Exception("output_variables must be a list")
            
            if any(var.lower() == 'all' for var in self.output_variables):
                print("outputting all variables")
                self.output_variables = self.owner.ordering
            else:
                for var in self.output_variables:
                    err_vars = []
                    if var not in ["all"]+self.owner.ordering:
                        err_vars.append(var)
                    if len(err_vars)>0:
                        raise Exception("output_variables must be a subset of the following: ",["all"]+self.owner.ordering)
                print("outputting only variables: ",self.output_variables)
        except Exception as e: 
            print("failes to load output variables from json file at location: ",kwargs["output_variables"])
            print(e)

    def write(self, output,check_nans, template, step,precip_output,**kwargs):
        for k, fs in enumerate(template):
            print("k: ",k," fs: ",fs)
            self.output.write(
                output[k, ...], check_nans=check_nans, template=fs, step=step
            )
            # break
            # if k == 4:
            #     break
        # if precip_output is not None:
        #     self.output.write(
        #         precip_output.squeeze(), check_nans=check_nans, template=template.sel(param="2t")[0], step=step, #param="tp",stepType="accum"
        #     )
    
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

        try:
            with open(kwargs["output_variables"]) as f:
                self.output_variables = json.load(f)
            
            if type(self.output_variables) != list:
                raise Exception("output_variables must be a list")
            
            if any(var.lower() == 'all' for var in self.output_variables):
                print("outputting all variables")
                self.output_variables = self.owner.ordering
            else:
                for var in self.output_variables:
                    err_vars = []
                    if var not in ["all"]+self.owner.ordering:
                        err_vars.append(var)
                    if len(err_vars)>0:
                        raise Exception("output_variables must be a subset of the following: ",["all"]+self.owner.ordering)
                print("outputting only variables: ",self.output_variables)
        except Exception as e: 
            print("failes to load output variables from json file at location: ",kwargs["output_variables"])
            print(e)
            exit(0)

        # self.dataset = None

    def write(self, output,check_nans, template,step,param_level_pl,param_sfc,precip_output):
        # copy input data (template) once to copy metadata into output dataset
        # if self.dataset is None:
        # works if template has equal size in every coordinate, won't work e.g. for fourcastnet where a few variables don't exist on every pressure level
        # x = template.to_xarray()
        # self.dataset = xr.zeros_like(template.to_xarray())
        if self.output_variables == self.owner.ordering and self.owner.model == 'sfno':
            print("hi")
        data_dict={}
        for idx,out_var in enumerate(self.owner.ordering):
            if out_var in self.output_variables:
                data_dict[out_var] = xr.DataArray(output[idx],dims=["latitude","longitude"],coords=dict(
                    latitude=(["latitude", "longitude"], np.tile(np.arange(-90,90.25,0.25)[::-1],(1440,1)).T),
                    longitude=(["latitude", "longitude"], np.tile(np.arange(0,360,0.25),(721,1))),
                ))

        if precip_output is not None:
            data_dict['tp'] = xr.DataArray(precip_output.squeeze(),dims=["latitude","longitude"],coords=dict(
                    latitude=(["latitude", "longitude"], np.tile(np.arange(-90,90.25,0.25)[::-1],(1440,1)).T),
                    longitude=(["latitude", "longitude"], np.tile(np.arange(0,360,0.25),(721,1))),
                ))
            self.dataset = self.dataset.assign(tp=precip_output.squeeze())

        dataset = xr.Dataset(data_vars=data_dict,coords=dict(
                    latitude=(["latitude"], np.arange(-90,90.25,0.25)[::-1]),
                    longitude=([ "longitude"], np.arange(0,360,0.25)),
                    step=[np.timedelta64(step*60*60*10**9, 'ns')]
                ))
        
            # for variable in self.levels_per_pl:
            #     for level in self.levels_per_pl[variable]:
            #         var_name = variable + str(level)
            #         if var_name in self.output_variables:
            #             data_dict[var_name] = xr.DataArray(np.zeros_like(output[0]),dims=["time","latitude","longitude"],coords=dict())


                # if variable in self.ordering:
                #     self.dataset[variable] = xr.zeros_like(template.sel(param=variable).to_xarray())
                # self.dataset.assign(variable=xr.zeros_like(template.sel(param=variable).to_xarray()))

    #key present and new value is different

            # if not any(var.lower() == 'all' for var in self.output_variables):
            #     # delete variables that shouldn't be outputted
            #     self.dataset.drop(([var for var in self.owner.ordering if var not in self.output_variables]))
        
        # loop through variables and preassure levels and overwrite dataset with output from  model
        # k = 0
        # for sfc in param_sfc:
        #     # skip variables that shouldn't be outputted
        #     if sfc in self.output_variables:
        #         continue
        #     axistupel = tuple(range(len(self.dataset[sfc].shape) - 2))
        #     self.dataset[sfc].values = np.expand_dims(output[k],axis=axistupel)
        #     k += 1
        # pls, levels = param_level_pl
        # levels.reverse()
        # for pl in pls:
        #     for level in levels:
        #         # skip variables that shouldn't be outputted
        #         if sfc in self.output_variables:
        #             continue
        #         axistupel = tuple(range(len(self.dataset[pl].shape) - 3))
        #         self.dataset[pl].sel(isobaricInhPa=level).values = np.expand_dims(output[k],axis=axistupel)
        #         k += 1

        # if precip_output is not None:
        #     self.dataset['tp'].values = xr.DataArray(precip_output.squeeze(),attr={'GRIB_cfName':'total_precipitation'})
        #     print('test')
        #     self.dataset = self.dataset.assign(tp=precip_output.squeeze())

        # self.dataset = self.dataset.assign_coords(step=[np.timedelta64(step*60*60*10**9, 'ns')])
        return dataset.to_netcdf(os.path.join(self.subdir, self.pathString + '_step_'+str(step)+'.nc'))



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