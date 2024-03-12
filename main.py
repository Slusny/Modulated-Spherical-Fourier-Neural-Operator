# (C) Copyright 2023 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import argparse
import logging
import os
import shlex
import sys
import pdb
from pathlib import Path
import time
import wandb

import torch

# to get eccodes working on Ubuntu 20.04
# os.environ["LD_PRELOAD"] = '/usr/lib/x86_64-linux-gnu/libffi.so.7'
# in shell : export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libffi.so.7
os.putenv("LD_PRELOAD", "/usr/lib/x86_64-linux-gnu/libffi.so.7")
import ecmwflibs
import cfgrib

#if shipped as module, include in S2S_on_SFNO and remove absolute import to relative .inputs etc. . Also move main inside -> __main__.py
from S2S_on_SFNO.inputs import available_inputs
from S2S_on_SFNO.Models.models import Timer, available_models, load_model #########
from S2S_on_SFNO.outputs import available_outputs

LOG = logging.getLogger(__name__)

print("cuda available? : ",torch.cuda.is_available(),flush=True)

def _main():
    parser = argparse.ArgumentParser()

    # See https://github.com/pytorch/pytorch/issues/77764
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    
    parser.add_argument(
            "--test",
            action="store_true",
            help="execute test code",
        )
    parser.add_argument(
        "--model",
        action="store",
        required=True,
        choices=available_models(),
        help="Specify the model to run",
    )
    parser.add_argument(
        "--model-version",
        default="latest",
        help="Model versions: \n    SFNO: 0, film\n    Fourcastnet: 0, 1",
    )
    parser.add_argument(
        "--assets",
        action="store",
        help="Absolute path to directory containing the weights and other assets of the Model. \
              Model Name gets appended to asset path. E.g. /path/to/assets/{model}\
              Default behaviour is to load from assets-sub-directory.",
    )
    parser.add_argument(
        "--assets-film",
        action="store",
        help="Absolute path to directory containing the weights and other assets of the Film-Model.",
        default=None
    )
    parser.add_argument(
        "--assets-sub-directory",
        action='store',
        default="S2S_on_SFNO/Assets",
        help="Load assets from a subdirectory of this module based on the name of the model. \
              Defaults to ./S2S_on_SFNO/Assets/{model}. Gets overwritten by --assets",
    )
    parser.add_argument(
        "--path",
        help="Path where to write the output of the model. Default: S2S_on_SFNO/outputs/{model}",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=1,
        help="Number of threads. Only relevant for some models.",
    )
    parser.add_argument(
        "--only-gpu",
        help="Fail if GPU is not available",
        action="store_true",
    )
        # Metadata
    parser.add_argument(
        "--expver",
        help="Set the experiment version of the model output. Has higher priority than --metadata.",
    )
    parser.add_argument(
        "--class",
        help="Set the 'class' metadata of the model output. Has higher priority than --metadata.",
        metavar="CLASS",
        dest="class_",
    )
    parser.add_argument(
        "--metadata",
        help="Set additional metadata metadata in the model output",
        metavar="KEY=VALUE",
        action="append",
    )
    parser.add_argument(
        "--model-args",
        help="specific model arguments for initialization of the model",
        action="store",
        default=None,
    )
        # Utilities
    parser.add_argument(
        "--assets-list",
        help="List the assets used by the model",
        action="store_true",
    )
    parser.add_argument(
        "--fields",
        help="Show the fields needed as input for the model",
        action="store_true",
    )

    # Data
    data = parser.add_argument_group('Data and Data Sources')
    data.add_argument(
        "--download-assets",
        help="Download assets (weights and means/std from ecmwf for pretrained models) if they do not exists.",
        action="store_true",
    )
    data.add_argument(
        "--input",
        default="cds",
        help="Source to use for downloading data, local file or cluster(uni tuebingen) specific data structure",
        choices=available_inputs(),
    )
    data.add_argument(
        "--input-store",
        default=None,
        help="If you download data from cds or mars and want to store it somewhere else, specify a path here. Default behaviour is to only temporary cache the data. The name of the file will be ClimateInputData_{YYYYMMDDHH}.grib at specified path",
        action="store"
    )
    data.add_argument(
        "--file",
        help="Specify path to file with input weather data. Sets source=file automatically",
    )
    data.add_argument(
        "--era5-path",
        default="/mnt/qb/goswami/data/era5",
        help="path to era5 data when using input=localERA5",
    )
        # Mars requests options
    data.add_argument(
        "--archive-requests",
        help=(
            "Save mars archive requests to FILE. Legacy option, only for mars requests."
            "Use --requests-extra to extend or overide the requests. "
        ),
        metavar="FILE",
    )
    data.add_argument(
        "--retrieve-requests",
        help=(
            "Print mars requests to stdout."
            "Use --requests-extra to extend or overide the requests. "
        ),
        action="store_true",
    )
    data.add_argument(
        "--requests-extra",
        help=(
            "Extends the retrieve or archive requests with a list of key1=value1,key2=value."
        ),
    )
    data.add_argument(
        "--json",
        action="store_true",
        help=("Dump the requests in JSON format."),
    )

    # Running
    running = parser.add_argument_group('Inference Parameters')
    running.add_argument(
        "--lead-time",
        type=int,
        default=240,
        help="Length of forecast in hours.",
    )
    running.add_argument(
        "--date",
        default="-1",
        help="For which analysis date to start the inference (default: -1 = yesterday). Format: YYYYMMDD",
    )
    running.add_argument(
        "--time",
        type=int,
        default=12,
        help="For which analysis time to start the inference (default: 12). Format: HHMM",
    )
    running.add_argument(
        "--output",
        default="grib",
        help="choose output format. Default: grib",
        choices=available_outputs(),
    )
    running.add_argument(
        "--output-variables",
        default="./S2S_on_SFNO/outputs/output-variables.json",
        help="Specify path to a json file detailing which variables to output. Default: all.",
    )
    running.add_argument(
        "--dump-provenance",
        action="store_true",
        help=("Dump information for tracking provenance."),
    )
    running.add_argument(
        "--hindcast-reference-year",
        help="For encoding hincast-like outputs",
    )
    running.add_argument(
        "--staging-dates",
        help="For encoding hincast-like outputs",
    )

    # Training
    training = parser.add_argument_group('Training Parameters')
    training.add_argument(
        "--train",
        help="train model",
        action="store_true",
    )
    training.add_argument(
        "--trainingset-start-year",
        help="specify training dataset by start year",
        action="store",
        default=1959,
        type=int
    )
    training.add_argument(
        "--trainingset-end-year",
        help="specify training dataset by start year",
        action="store",
        default=1975,
        type=int
    )
    training.add_argument(
        "--trainingdata-path",
        help="path to training data zarr file",
        action="store",
        default="/mnt/ceph/goswamicd/datasets/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr"
    )
    training.add_argument(
        "--training-workers",
        help="number of workers to use in dataloader for training",
        action="store",
        default=1,
        type=int
    )
    training.add_argument(
        "--batch-size",
        action="store",
        default=4,
        type=int
    )

    # Logging
    logging_parser = parser.add_argument_group('Logging')
    logging_parser.add_argument(
        "--debug",
        action="store_true",
        help="Turn debugger on (pdb).",
    )
    logging_parser.add_argument(
        '--wandb', 
        action='store_true',
        help='use weights and biases'
    )
    logging_parser.add_argument(
        '--wandb_resume', 
        action='store', 
        default=None,             
        type=str, 
        help='resume existing weights and biases run')


    args, unknownargs = parser.parse_known_args()

    # Format Assets path
    if args.assets:
        args.assets = os.path.join(os.path.abspath(args.assets),args.model)
    elif args.assets_sub_directory:
        args.assets = os.path.join(Path(".").absolute(),args.assets_sub_directory,args.model)

    # Format Output path
    timestr = time.strftime("%Y%m%dT%H%M")
    save_string = "leadtime_"+str(args.lead_time)+"_startDate_"+str(args.date)+str(args.time) +"_createdOn_"+timestr
    if args.path is None:
        outputDirPath = os.path.join(Path(".").absolute(),"S2S_on_SFNO/outputs",args.model)
    else:
        outputDirPath = os.path.join(args.path,args.model)
    
    args.path  = os.path.join(outputDirPath,save_string+".grib")
    if not os.path.exists(args.path):
        os.makedirs(os.path.dirname(args.path), exist_ok=True)

    if args.file is not None:
        args.input = "file"

    if not args.fields and not args.retrieve_requests:
        logging.basicConfig(
            level="DEBUG" if args.debug else "INFO",
            format="%(asctime)s %(levelname)s %(message)s",
        )
    
    if args.debug: #new
        pdb.set_trace()
        print("starting debugger")

    if args.metadata is None:
        args.metadata = []

    if args.expver is not None:
        args.metadata["expver"] = args.expver

    if args.class_ is not None:
        args.metadata["class"] = args.class_

    # Manipulation on args
    args.metadata = dict(kv.split("=") for kv in args.metadata)
    
    model = load_model(args.model, vars(args))

    if args.fields:
        model.print_fields()
        sys.exit(0)

    if args.requests_extra:
        if not args.retrieve_requests and not args.archive_requests:
            parser.error(
                "You need to specify --retrieve-requests or --archive-requests"
            )
            

    # This logic is a bit convoluted, but it is for backwards compatibility.
    if args.retrieve_requests or (args.requests_extra and not args.archive_requests):
        model.print_requests()
        sys.exit(0)

    if args.assets_list:
        model.print_assets_list()
        sys.exit(0)

    if args.test:
        from S2S_on_SFNO.Models.train import train
        train(vars(args))
        print("Test passed")
        sys.exit(0)
        
    if args.wandb   : 
        config_wandb = vars(args).copy()
        for key in ['notes','tags','wandb']:del config_wandb[key]
        del config_wandb
        if args.wandb_resume is not None :
            wandb_run = wandb.init(project=args.model + " - " +args.model_version, 
                config=args,
                notes=args.notes,
                tags=args.tags,
                resume="must",
                id=args.wandb_resume)
        else:
            wandb_run = wandb.init(project=args.model + " - " +args.model_version, 
                config=args,
                notes=args.notes,
                tags=args.tags)
    else : wandb_run = None


    if args.train:
        kwargs = vars(args)
        model.training(**kwargs,wandb_run=wandb_run)
    else:

        try:
            model.run()
        except FileNotFoundError as e:
            LOG.exception(e)
            LOG.error(
                "It is possible that some files requited by %s are missing.\
                \n    Or that the assets path is not set correctly.",
                args.model,
            )
            LOG.error("Rerun the command as:")
            LOG.error(
                "   %s",
                shlex.join([sys.argv[0], "--download-assets"] + sys.argv[1:]), ## download assets call not nessessary
            )
            sys.exit(1)

    model.finalise()

    if args.dump_provenance:
        with Timer("Collect provenance information"):
            file = os.path.join(outputDirPath,save_string + "_provenance.json")
            with open(file, "w") as f:
                prov = model.provenance()
                import json  # import here so it is not listed in provenance
                json.dump(prov, f, indent=4)


def main():
    with Timer("Total time"):
        _main()


if __name__ == "__main__":
    # args = ["--model","sfno","--test","--training-workers","0","--batch-size","1","--debug"]
    # for arg in args: sys.argv.append(arg)
    main()
