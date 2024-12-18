"""
File: main.py
Author: Lennart Slusny
Date: June 28, 2024
Description: Execute this script in the console to access and run 
    the core features of the project through the command-line interface. 
    Use the --help option for detailed information on available commands.
"""
import argparse
import logging
import os
import shlex
import sys
import pdb
from pathlib import Path
import time
import traceback
import torch
import glob
import re
import datetime
import math
import dask

from MSFNO.inputs import available_inputs
from MSFNO.Models.models import load_model 
from MSFNO.utils import Timer
from MSFNO.outputs import available_outputs
from MSFNO.Models.train import Trainer
from MSFNO.utils import FinTraining

import torch.multiprocessing as mp
from torch.distributed import init_process_group, destroy_process_group

# global variables
os.environ["WANDB__SERVICE_WAIT"] = "300"

# setup processes for distributed training
def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "31350"
    os.environ["WANDB__SERVICE_WAIT"] = "300"
    torch.cuda.set_device(rank)
    init_process_group(backend="nccl", rank=rank, world_size=world_size, timeout=datetime.timedelta(hours=2))

LOG = logging.getLogger(__name__)
LOG.addHandler(logging.StreamHandler(sys.stdout)) 
cuda_available = torch.cuda.is_available()
print("cuda available? : ",cuda_available,flush=True)

def main(rank=0,args={},arg_groups={},world_size=1):
    
    # to speed up prallel loading of data
    dask.config.set(scheduler='synchronous')

    args.rank = rank
    args.world_size = world_size

    # logging Config
    if args.log_file:logging.basicConfig(level=logging.INFO, filename=args.log_file,filemode="a")
    if args.jobID is not None: LOG.info("Slurm Job ID: %s", args.jobID)

    # start python debugger, disable multiprocess features
    if args.debug: 
        if not args.ddp: pdb.set_trace()
        torch.autograd.set_detect_anomaly(True)
        args.training_workers = 0
        if args.rank == 0:
            print("starting debugger",flush=True) 
            print("setting training workers to 0 to be able to debug code in ",flush=True)

    if args.ddp:
        print("rank ",rank,flush=True)
        ddp_setup(rank,world_size)

    # Format Assets path
    if args.assets_sub_directory:
        args.assets = os.path.join(Path(".").absolute(),args.assets_sub_directory)


    if args.requests_extra:
        if not args.retrieve_requests and not args.archive_requests:
            parser.error(
                "You need to specify --retrieve-requests or --archive-requests"
            )

    if not args.fields and not args.retrieve_requests:
        logging.basicConfig(
            level="DEBUG" if args.debug else "INFO",
            format="%(asctime)s %(levelname)s %(message)s",
        )

    if args.file is not None:
        args.input = "file"

    if args.metadata is None:
        args.metadata = {}

    if args.expver is not None:
        args.metadata["expver"] = args.expver

    if args.class_ is not None:
        args.metadata["class"] = args.class_


    # Manipulation on args
    if args.metadata: args.metadata = dict(kv.split("=") for kv in args.metadata)

    # Add extra steps to multi_step_training if we want to skip steps during training
    if args.training_step_skip > 0:
        if args.multi_step_training > 0:
            args.multi_step_training = args.multi_step_training + args.training_step_skip*(args.multi_step_training)
        else:
            print("multi-step-skip given but no multi-step-training = 0. Specify the number of steps in multi-step-training larger 0.",flush=True)
    if args.validation_step_skip > 0:
        if args.multi_step_validation > 0:
            args.multi_step_validation = args.multi_step_validation + args.validation_step_skip*(args.multi_step_validation)
        else:
            print("multi-step-skip given but no multi-step-validation = 0. Specify the number of steps in multi-step-validation larger 0.",flush=True)

    if args.batch_size_validation is None:
        args.batch_size_validation = args.batch_size

    # set film_gen_type if model version film is selected but no generator to default value
    if args.film_gen_type:
        if args.film_gen_type.lower() == "none" : args.film_gen_type = None
    if args.model_version == "film" and args.film_gen_type is None: 
        print("using film generator: gcn_custom",flush=True)
        args.film_gen_type = "gcn_custom"
    # scheduler is updated in every validation interval. To arive at the total horizon in standard iters we divide by the validation interval
    args.scheduler_horizon = math.floor(args.scheduler_horizon/(abs(args.validation_interval)*args.batch_size*(args.accumulation_steps+1)*args.world_size))

    # Format Output path
    timestr = time.strftime("%Y%m%dT%H%M")
    # save_string to save output data if model.run is called (only for runs not for training)
    if args.path is None:
        outputDirPath = os.path.join(Path(".").absolute(),"S2S_on_SFNO/outputs",args.model_type)
        if args.model_version == "film":
            outputDirPath = os.path.join(outputDirPath,"film-"+args.film_gen_type)
    else:
        outputDirPath = args.path
    args.path  = outputDirPath

    if args.time_limit is not None:
        time_arr = args.time_limit.split("-") 
        seconds = 0.
        if len(time_arr) == 2:
            seconds += int(time_arr[0])*24*3600
        x = time.strptime(time_arr[-1], "%H:%M")
        seconds += datetime.timedelta(hours=x.tm_hour,minutes=x.tm_min,seconds=x.tm_sec).total_seconds()
        args.time_limit = seconds

    
    # timestring for logging and saveing purposes
    args.timestr = timestr
    if not os.path.exists(args.path):
        os.makedirs(os.path.dirname(args.path), exist_ok=True)

    # Print args
    if rank == 0:
        print("Script called with the following parameters:")
        for group,value in arg_groups.items():
            if group == 'positional arguments': continue
            print(" --",group)
            for k,v in sorted(vars(value).items()):
                print("    ",k," : ",v)
            print("")

    # Load parameters from checkpoint if given and load model
    resume_cp = args.resume_checkpoint
    map_location=torch.device(rank) if cuda_available else 'cpu'
    if args.eval_model:
        resume_cp = list(sorted(glob.glob(os.path.join(args.eval_checkpoint_path,"checkpoint_*")),key=len))[-1]
    if resume_cp:
        cp = torch.load(resume_cp,map_location=map_location)
        if not 'hyperparameters' in cp.keys(): 
            print("couldn't load model configuration from checkpoint")
            model = load_model(args.model_type, vars(args))
        else:
            model_args = cp["hyperparameters"].copy()

            # overwrite checkpoint parameters with given parameters, attention! : ignores default values, only specified ones
            
            for passed_arg in sys.argv[1:]:
                if passed_arg.startswith("--"):
                    dest = next(x for x in parser._actions if x.option_strings[0] == passed_arg).dest
                    # skip Architectural changes
                    if dest in (list(vars(arg_groups["Architecture"]).keys())+list(vars(arg_groups["Architecture Film Gen"]).keys())): continue # do we want to skip Architecture as well?
                    # modify parameters if an other is given:
                    if dest == "batch_size":
                        model_args["batch_size_validation"] =  vars(args)["batch_size_validation"]
                    model_args[dest] = vars(args)[dest]

            # set flags back to default
            for k,v in vars(args).items():
                if k in ['train','timestamp','wandb','wandb_resume','wandb_project','ddp','enable_amp','time_limit','set_epoch','debug','rank','world_size','no_wandb','resume_optimizer','resume_scheduler','no-scratch']: 
                    if k == 'enable_amp' and args.train: continue
                    model_args[k] = v

            # copy parameters present in current version, but not in checkpoint
            for k,v in vars(args).items():
                if k not in model_args.keys():
                    model_args[k] = v

            # copy film architecture hyperparameters if different film-layer is given
            if args.film_weights:
                film_cp = torch.load(args.film_weights,map_location=map_location)
                if not 'hyperparameters' in cp.keys(): 
                    print("couldn't load film model configuration from checkpoint")
                else:
                    for k,v in film_cp["hyperparameters"].items():
                        if k in vars(arg_groups["Architecture Film Gen"]).keys(): 
                            model_args[k] = v
                del film_cp
            del cp
            if rank == 0:
                print("\nScript updated with Checkpoint parameters to:")
                for k,v in sorted(model_args.items()):
                    print("    ",k," : ",v)
            kwargs = model_args
            if args.no_wandb: model_args["wandb"] = False
            model = load_model(model_args["model_type"], kwargs)
            # trainer is still called with the original args not model_args but model_args should only modify architecture - do it nevertheless
    else:
        # if only film weights are given, load film model
        kwargs = vars(args)
        if args.film_weights:
            film_cp = torch.load(args.film_weights,map_location=map_location)
            if not 'hyperparameters' in film_cp.keys(): 
                print("couldn't load film model configuration from checkpoint")
            else:
                for k,v in film_cp["hyperparameters"].items():
                    if k in vars(arg_groups["Architecture Film Gen"]).keys(): 
                        kwargs[k] = v
            del film_cp

            if rank == 0:
                print("\nScript updated with Checkpoint parameters:")
                for k,v in sorted(kwargs.items()):
                    print("    ",k," : ",v)
        model = load_model(args.model_type, kwargs)

    if args.fields:
        model.print_fields()
        return

    # This logic is a bit convoluted, but it is needed for backwards compatibility.
    if args.retrieve_requests or (args.requests_extra and not args.archive_requests):
        model.print_requests()
        return

    if args.assets_list:
        model.print_assets_list()
        return

    elif args.train:

        print("")
        print("Started training ")
        LOG.info("Process (rank="+str(rank)+") ID: %s", os.getpid())


        trainer = Trainer(model,kwargs)
        try:
            trainer.train()
        except FinTraining as e:
            print("Training finished due to",e)
        except :
            LOG.error(traceback.format_exc())
            print("shutting down training")
            trainer.finalise()

    elif args.test_performance:
        trainer = Trainer(model,kwargs)
        trainer.test_performance()
        return

    elif args.test_dataloader_speed:
        trainer = Trainer(model,kwargs)
        trainer.test_dataloader_speed()
        return

    elif args.test_batch_size:
        trainer = Trainer(model,kwargs)
        trainer.test_batch_size()
        return

    elif args.save_data:
        trainer = Trainer(model,kwargs)
        trainer.save_data()
        return

    elif args.save_forecast:
        trainer = Trainer(model,kwargs)
        trainer.save_forecast()
        return

    elif args.eval_model:
        print("evaluating models")
        checkpoint_list = list(sorted(glob.glob(os.path.join(args.eval_checkpoint_path,"checkpoint_*")),key=len)) 

        # select equidistance checkpoints from all checkpoints
        if len(args.eval_checkpoints) > 0:
            checkpoint_file = re.sub(r"\d+","{}",checkpoint_list[-1].split("/")[-1])
            checkpoint_list_shorten = [os.path.join(args.eval_checkpoint_path,checkpoint_file.format(checkpoint)) for checkpoint in args.eval_checkpoints]
        elif args.eval_checkpoint_num > 1:
            num_checkpoints = len(checkpoint_list)
            checkpoint_list_shorten = checkpoint_list[::num_checkpoints//args.eval_checkpoint_num]
            if len(checkpoint_list_shorten)>args.eval_checkpoint_num:
                checkpoint_list_shorten[-1] = checkpoint_list[-1]
            else:
                checkpoint_list_shorten.append(checkpoint_list[-1])
            checkpoint_list_shorten.pop(0)
        elif args.eval_checkpoint_num == -1:
            checkpoint_list_shorten = checkpoint_list
        else:
            checkpoint_list_shorten = [checkpoint_list[-1]]#,checkpoint_list[2],checkpoint_list[-1]]
        print("loading :")
        for cp in checkpoint_list_shorten:print("    "+cp)
        # #sfno
        # sfno_kwargs = vars(args)
        # sfno_kwargs["model_version"] = "release"
        # sfno = load_model('sfno', sfno_kwargs)

        #save folder
        save_path = os.path.join(args.eval_checkpoint_path,"figures","valid_iter"+str(args.validation_epochs))#str(args.multi_step_validation//(args.validation_step_skip+1))
        os.makedirs(save_path, exist_ok=True)
        os.makedirs(os.path.join(save_path,"variable_plots"), exist_ok=True)

        # model.evaluate_model(checkpoint_list_shorten,save_path)
        trainer = Trainer(model,kwargs)
        trainer.evaluate_model(checkpoint_list_shorten,save_path)

    elif args.run:

        try:
            model.running()
        except FileNotFoundError as e:
            LOG.exception(e)
            LOG.error(
                "It is possible that some files requited by %s are missing.\
                \n    Or that the assets path is not set correctly.",
                args.model_type,
            )
            LOG.error("Rerun the command as:")
            LOG.error(
                "   %s",
                shlex.join([sys.argv[0], "--download-assets"] + sys.argv[1:]), ## download assets call not nessessary
            )
            if kwargs["model_type"] == "mae":
                model.save_cls()
                return
    else:
        print("No action specified (--train or --run). Exiting.")
    model.finalise()

    if args.dump_provenance:
        with Timer("Collect provenance information"):
            file = os.path.join(outputDirPath,save_string + "_provenance.json")
            with open(file, "w") as f:
                prov = model.provenance()
                import json  # import here so it is not listed in provenance
                json.dump(prov, f, indent=4)


def return_trainer(args):
    print("returning trainer")
    global do_return_trainer 
    do_return_trainer = True
    # args = ["--model","sfno","--test","--training-workers","0","--batch-size","1","--debug"]
    sys.argv = [sys.argv[0]]
    for arg in args: sys.argv.append(arg)
    return main()



# Commnand Line Interface

parser = argparse.ArgumentParser()

# See https://github.com/pytorch/pytorch/issues/77764
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

parser.add_argument(
    "--test",
    action="store_true",
    help="execute test code",
    )
parser.add_argument(
    "--assets",
    action="store",
    help="Absolute path to directory containing the weights and other assets of the Model. \
            Model Name gets appended to asset path. E.g. /path/to/assets/{model}\
            Default behaviour is to load from assets-sub-directory.",
    default="/mnt/qb/work2/goswami0/gkd965/Assets"
)
parser.add_argument(
    "--film-weights",
    action="store",
    help="Absolute path to weights.tar file containing the weights of the Film-Model.",
    default=None,
)
parser.add_argument(
    "--sfno-weights",
    action="store",
    help="Absolute path to weights.tar file containing the weights of the SFNO-Model.",
    default=None,
)
parser.add_argument(
    "--assets-sub-directory",
    action='store',
    default=None,
    # help="Load assets from a subdirectory of this module based on the name of the model. \
    #       Defaults to ./S2S_on_SFNO/Assets/{model}. Gets overwritten by --assets",
)
parser.add_argument(
    "--output-path",
    help="Path where to write the output of the model if it is run (atmospheric fields in grib or netcdf format). For training data output (e.g. checkpoints) look for save-path. Default ouput-path: S2S_on_SFNO/outputs/{model}",
    default="/mnt/qb/work2/goswami0/gkd965/outputs",
    dest="path",
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
parser.add_argument(
    "--cpu",
    help="Use CPU",
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
parser.add_argument(
    "--jobID",
    help="log the slurm JobID",
    action="store",
    default=None,
)
parser.add_argument(
    "--save-data",
    action="store_true",
)

parser.add_argument(
    "--time-limit",
    action="store",
    default=None,
    help="time limit in slurm, to save a model before it ends",
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
    default="none",
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
data.add_argument(
    "--coarse-level",
    action="store",
    default=4,
    help=("factor by which the sst data gets reduced in lat, long dimensions"),
)
data.add_argument(
    "--cls",
    action="store",
    default=None,
    help=("path to numpy file containing cls tokens from MAE model"),
)
data.add_argument(
    "--oni-path",
    action="store",
    default=None,
    help=("path to numpy file containing oni indices"),
)
data.add_argument(
    "--past-sst",
    action="store_true",
    help=("by default the sst data is taken from the future. \
            If conditioned on a single on a single sst image it is from the day for wich a forcast is done. \
            In the case of the MAE the sst start from the input date to #temporal-steps into the futre.\
            By setting this flag the sst data is taken from the past."),
)
data.add_argument(
    "--oni",
    action="store_true",
    help=("calculate and use the ONI index as ground truth"),
)

data.add_argument(
    "--no-shuffle",
    action="store_true",
    help=("dont use shuffle in dataloader"),
)

# Running
running = parser.add_argument_group('Inference Parameters')
running.add_argument(
    "--run",
    help="run model",
    action="store_true",
)
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
    default=1979,
    type=int
)
training.add_argument(
    "--trainingset-end-year",
    help="specify training dataset by end year. No dates from the end year specified and later will be used.",
    action="store",
    default=2016,
    type=int
)
training.add_argument(
    "--validationset-start-year",
    help="specify validation dataset by start year",
    action="store",
    default=2016,
    type=int
)
training.add_argument(
    "--validationset-end-year",
    help="specify validation dataset by end year. No dates from the end year specified and later will be used.",
    action="store",
    default=2018,
    type=int
)
training.add_argument(
    "--validation-interval",
    help="after running ... expochs, validate the model",
    action="store",
    default=150,
    type=int
)
training.add_argument(
    "--save-checkpoint-interval",
    help="saving every x validation. E.g. if validation-intervall is 100 and save-checkpoint-interval is 10, the model is saved every 1000 iterations",
    action="store",
    default=10,
    type=int
)
training.add_argument(
    "--validation-epochs",
    help="over how many epochs should be validated",
    action="store",
    default=20,
    type=int
)
training.add_argument(
    "--training-epochs",
    help="over how many epochs should be trained",
    action="store",
    default=20,
    type=int
)
training.add_argument(
    "--multi-step-validation",
    help="how many consecutive datapoints should be loaded to used to calculate an autoregressive validation loss ",
    action="store",
    default=0,
    type=int
)
training.add_argument(
    "--multi-step-training",
    help="calculate loss over multiple autoregressive prediction steps",
    default=0,
    type=int,
    action="store",
)
training.add_argument(
    "--training-step-skip",
    help="skip the x amount of autoregressive steps in the multi-step training to calculate the loss",
    default=0,
    type=int,
    action="store",
)
training.add_argument(
    "--accumulation-steps",
    help="accumulate gradients over x steps. Increases batch size by withoutincreasing memory consumption",
    default=0,
    type=int,
    action="store",
)
training.add_argument(
    "--validation-step-skip",
    help="skip the x amount of autoregressive steps in the multi-step validation to calculate the loss",
    default=0,
    type=int,
    action="store",
)
training.add_argument(
    "--val-loss-threshold",
    help="increasing the scaleing of the film layer based on the validation loss. If the validation loss is lower than this threshold, the scaleing is increased by 0.05",
    action="store",
    default=0.4,
    type=float
)
training.add_argument(
    "--scaling-horizon",
    help="how many steps should it take to reach scale=1",
    action="store",
    default=2000,
    type=float
)
training.add_argument(
    "--trainingdata-path",
    help="path to training data zarr file",
    action="store",
    default="/mnt/qb/goswami/data/era5/weatherbench2/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr"
)
training.add_argument(
    "--trainingdata-u100-path",
    help="path to training data zarr file for u100m",
    action="store",
    default="/mnt/qb/goswami/data/era5/u100m_v100m_721x1440/u100m_1959-2022_721x1440_correct_chunk_new_mean_INTERPOLATE.zarr"
)
training.add_argument(
    "--trainingdata-v100-path",
    help="path to training data zarr file for v100m",
    action="store",
    default="/mnt/qb/goswami/data/era5/u100m_v100m_721x1440/v100m_1959-2023-10_721x1440_correct_chunk_new_mean_INTERPOLATE.zarr"
)
training.add_argument(
    "--training-workers",
    help="number of workers to use in dataloader for training",
    action="store",
    default=6,
    type=int
)
training.add_argument(
    "--batch-size",
    action="store",
    default=1,
    type=int
)
training.add_argument(
    "--batch-size-validation",
    action="store",
    default=None,
    type=int
)
# FourCastNet uses 0.0005
training.add_argument(
    "--learning-rate",
    action="store",
    default=0.001,#2
    type=float
)
# FourCastNet uses Cosine
training.add_argument(
    "--scheduler",
    action="store",
    default="None",
    help="which pytorch scheduler to use",
    dest="scheduler_type",
    type=str
)
training.add_argument(
    "--scheduler-horizon",
    action="store",
    default=2000,
    help="defines the horizon on which the scheduler should reset the learning rate. In case of CosineAnnealingWarmRestarts this modifies the parameter T_0",
    type=int
)
training.add_argument(
    "--save-path",
    action="store",
    default="/mnt/qb/work2/goswami0/gkd965/checkpoints",
    type=str,
    help="path to save checkpoints and training data, not used for running the model"
)
training.add_argument(
    "--checkpointing-mlp",
    action="store_true",
    help="Trades compute for memory. Checkpoints MLPs in SFNO (encoder,decoder and MLP in SFNO-Block). Only partly computes the forward path and recomputes missing parts during backward pass. See pytroch checkpointing. Needed to perform multistep training. pure sfno alone already consumes 28GB VRAM"
)
training.add_argument(
    "--checkpointing-block",
    action="store_true",
    help="Trades compute for memory. Checkpoints SFNO-Block. Only partly computes the forward path and recomputes missing parts during backward pass. See pytroch checkpointing. Needed to perform multistep training. pure sfno alone already consumes 28GB VRAM"
)
training.add_argument(
    "--checkpointing-encoder",
    action="store_true",
    help="Trades compute for memory. Checkpoints SFNO-encoder. Only partly computes the forward path and recomputes missing parts during backward pass. See pytroch checkpointing. Needed to perform multistep training. pure sfno alone already consumes 28GB VRAM"
)
training.add_argument(
    "--checkpointing-decoder",
    action="store_true",
    help="Trades compute for memory. Checkpoints SFNO-decoder. Only partly computes the forward path and recomputes missing parts during backward pass. See pytroch checkpointing. Needed to perform multistep training. pure sfno alone already consumes 28GB VRAM"
)
training.add_argument(
    "--resume-checkpoint",
    action="store",
    default=None,
    help="Load model from checkpoint and use its configuration to initialize the model"
)
training.add_argument(
    "--no-pretrained-sfno",
    action="store_true",
    help="Use pretrained sfno model from ecmwf"
)
training.add_argument(
    "--enable-amp",
    action="store_true",
    help="Save RAM with AMP"
)
training.add_argument(
    "--optimizer",
    action="store",
    default="Adam",
    help="Optimizer to use",
    choices=["Adam","SGD","LBFGS","AdamW"],
)
training.add_argument(
    "--weight-decay",
    action="store",
    default=0.001,
    help="",
    type=float,
)
training.add_argument(
    "--dropout",
    action="store",
    default=0.,
    help="",
    type=float,
)
training.add_argument(
    "--loss-fn",
    action="store",
    help="Which loss function to use",
    default="L2Sphere_noSine",
    choices=["MSE","CosineMSE","L2Sphere",'L2Sphere_noSine',"NormalCRPS","L1"],
)
training.add_argument(
    "--loss-reduction",
    action="store",
    help="Which loss reduction method to use",
    default="mean",
    choices=["mean","none","sum"],
)
training.add_argument(
    "--test-performance",
    action="store_true",
    help="run speed test for dataloader and model performance",
)
training.add_argument(
    "--test-dataloader-speed",
    action="store_true",
    help="run speed test for dataloader and model performance",
)
training.add_argument(
    "--test-batch-size",
    action="store_true",
    help="run speed test for dataloader and model performance",
)
training.add_argument(
    "--num-iterations",
    action="store",
    type=int,
    default=100,
    help="over how many iterations should the speed test be run",
)
training.add_argument(
    "--batch-size-step",
    action="store",
    type=int,
    default=1,
    help="when testing for an optimal batch size, how large should be the inital step size",
)
training.add_argument(
    "--save-forecast",
    action="store_true",
)
training.add_argument(
    "--ddp",
    action="store_true",
)
training.add_argument(
    "--world-size",
    action="store",
    type=int,
    default=None,
)
training.add_argument(
    "--set-rank",
    action="store",
    type=int,
    default=0,
)
training.add_argument(
    "--discount-factor",
    action="store",
    type=float,
    default=0.9,
)
training.add_argument(
    "--set-epoch",
    action="store",
    type=int,
    default=None,
    help="set the epoch to a specific integer. This is useful if training is resumed. The epoch is used as the seed for the dataloader",
)
training.add_argument(
    "--resume-scheduler",
    action="store_true",
)
training.add_argument(
    "--resume-optimizer",
    action="store_true",
)
training.add_argument(
    "--no-scratch",
    action="store_true",
)
training.add_argument(
    "--retrain-film",
    action="store_true",
)
training.add_argument(
    "--scale-weight",
    action="store",
    type=int,
    default=1,
    help="",
)



# Evaluation
evaluate = parser.add_argument_group('Evaluate Models')
evaluate.add_argument(
    "--eval-model",
    help="evaluate model list of checkpoints for autoregressive forecast",
    action="store_true",
)
evaluate.add_argument(
    "--eval-sfno",
    help="evaluate base sfno model",
    action="store_true",
)
evaluate.add_argument(
    "--eval-checkpoint-num",
    help="how many checkpoints should be evaluated from --eval-checkpoint-path. The checkpoints are selected equidistantly. -1 evaluates all checkpoints",
    action="store",
    type=int,
    default=1,
)
evaluate.add_argument(
    "--eval-checkpoints",
    help="Name the epoch for which checkpoints should be loaded. E.g. --eval-checkpoints 500 700 900",
    nargs='+',
    default=[],
)
evaluate.add_argument(
    "--eval-checkpoint-path",
    help="evaluate model list of checkpoints for autoregressive forecast",
    action="store",
    type=str
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
    '--no-wandb', 
    action='store_true',
    help='dont use weights and biases'
)
logging_parser.add_argument(
    '--wandb-resume', 
    action='store', 
    default=None,             
    type=str, 
    help='resume existing weights and biases run')

logging_parser.add_argument(
    '--notes', 
    action='store', 
    default=None,             
    type=str, 
    help='notes for wandb')

logging_parser.add_argument(
    '--wandb-project', 
    action='store', 
    default=None,             
    type=str, 
    help='which project to log to. if none given defaults to {model}-{model-version}')

logging_parser.add_argument(
    '--tags', 
    action='store', 
    default=None,             
    type=str, 
    help='tags for wandb')
logging_parser.add_argument(
    '--advanced-logging', 
    action='store_true',
    help='Log more values like the gamma, beta activations. Consumes more GPU memory.'
)
logging_parser.add_argument(
    '--log-file', 
    action='store',
    default=None,
    help='Log stdout/err to file (for module logging, some logs are printed so a redirect > log_file )'
)
# Architecture
architecture_parser = parser.add_argument_group('Architecture')
architecture_parser.add_argument(
    "--model",
    action="store",
    #choices=available_models(),
    choices=["sfno","fcn","mae"],
    dest="model_type",
    help="Specify the model to run"
)
architecture_parser.add_argument(
    "--model-version",
    default="latest",
    help="Model versions: \n    SFNO: [latest, film]\n    Fourcastnet: [0, 1]\n    MAE: [latest, lin-probe]",
)
architecture_parser.add_argument(
    "--film-gen",
    default=None,
    type=str,
    dest="film_gen_type",
    help="Which type of film generator to use in the filmed model.",
    choices=["none","gcn","gcn_custom","transformer","mae"]
)
architecture_film_parser = parser.add_argument_group('Architecture Film Gen')
architecture_film_parser.add_argument(
    '--film-layers', 
    action='store',
    type=int,
    default=1,
    help='How many sfno blocks should be modulated with a dedicated film layer. Default: 1',
)
architecture_film_parser.add_argument(
    '--model-depth', 
    action='store',
    type=int,
    default=6,
    help='Number of layers for film generator',
)
architecture_film_parser.add_argument(
    '--temporal-step', 
    action='store',
    type=int,
    default=28,
    help='How many 6 hr steps should be included in the temporal dimension for the mae model. Needs to be larger than 0',
)
architecture_film_parser.add_argument(
    '--nan-mask-threshold',  
    action='store',
    type=float,
    default=0.5,
    help='token with a ratio of nan values higher than this threshold are masked',
)
architecture_film_parser.add_argument(
    '--patch-size', 
    action='store',
    type=int,
    nargs="+",
    default=[28,9,9],
    help='Define the patch sizes for the MAE (temporal, lat, lon) and Transfomrer (lat,long)',
)
architecture_film_parser.add_argument(
    '--embed-dim', 
    action='store',
    type=int,
    default=512,
    help='',
)
architecture_film_parser.add_argument(
    '--mlp-dim', 
    action='store',
    type=int,
    default=1024,
    help='',
)
architecture_film_parser.add_argument(
    '--repeat-film', 
    action='store_true',
    help='repeat the same film modulation arcoss all sfno blocks',
)

if __name__ == "__main__":

    args = parser.parse_args()

    # get parameters split by groups
    arg_groups={}
    for group in parser._action_groups:
        group_dict={a.dest:getattr(args,a.dest,None) for a in group._group_actions}
        arg_groups[group.title]=argparse.Namespace(**group_dict)


    with Timer("Total time"):
        if args.ddp:
            if args.world_size is None:
                args.world_size = torch.cuda.device_count()
            mp.spawn(main, args=(args,arg_groups,args.world_size), nprocs=args.world_size, join=True)
            destroy_process_group()
        else:
            main(rank=args.set_rank,args=args,arg_groups=arg_groups)


