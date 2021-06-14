"""
Select model using gridsearch
"""

import argparse
import os
import pickle
import yaml
import torch

import numpy as np

from utils_prediction.dataloader.mimic4 import dataloader
from utils_prediction.model_selection import gridsearch_nn
from utils_prediction.nn.models import FixedWidthModel
from utils_prediction.utils import str2bool

#-------------------------------------------------------------------
# arg parser
#-------------------------------------------------------------------

# Init
parser = argparse.ArgumentParser(
    description = "Grid search hyperparameters for nn & learning algorithm"
    )

## required without defaults
# general
parser.add_argument(
    "--analysis_id", 
    type = str, 
    required = True, 
    help="task: ['mortality','longlos','sepsis','invasivevent']"
)

## with defaults
# dataset related
parser.add_argument(
    "--datasets_fpath",
    type = str,
    default = "/hpf/projects/lsung/projects/public/mimic4ds_public/mimic4ds/Experiments/domain_gen/artifacts",
    help = "path where datasets folder is stored"
    )

parser.add_argument(
    "--datasets_ftype",
    type = str,
    default = "parquet",
    help = "file type [parquet (default) or feather] of the datasets to be saved"
    )

parser.add_argument(
    "--loader_verbose",
    type = str2bool,
    default = "false",
    help = 'verbosity for dataloader [default: False]'
    )

# gridsearch related
parser.add_argument(
    "--grids_path",
    type = str,
    default = "/hpf/projects/lsung/projects/public/mimic4ds_public/mimic4ds/Experiments/domain_gen/hyperparams/nn_grid.yml",
    help = 'Path to file with hyperparameter grid.'
    )

parser.add_argument(
    "--save_fpath",
    type = str,
    default = "/hpf/projects/lsung/projects/mimpublic/mimic4ds_public/mimic4dsic4ds/Experiments/domain_gen/hyperparams/models/"
)

parser.add_argument(
    "--n_searches",
    type = int,
    default = 100,
    help = "number of total random searches [default 100]"
    )

parser.add_argument(
    "--seed",
    type = int,
    default = 44,
    help = "seed for deterministic training"
    )

#-------------------------------------------------------------------
# helper functions
#-------------------------------------------------------------------

# none

#-------------------------------------------------------------------
# run
#-------------------------------------------------------------------
if __name__ == "__main__":
    
    args = parser.parse_args()
    
    # set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    ## load datasets
    dataloader_config = {
        "analysis_id":args.analysis_id,
        "datasets_fpath":args.datasets_fpath,
        "datasets_ftype":args.datasets_ftype,
        "verbose":args.loader_verbose
    }
    print(f"loading datasets...")
    data = dataloader(**dataloader_config).load_datasets("preprocessed_dense")
    loaders = data.to_torch(group_var_name = 'group_var')
    
    ## Gridsearch
    # load grid
    print(f"loading hyper-parameter grid from {args.grids_path}")
    
    param_grid = yaml.load(
        open(args.grids_path,'r'), 
        Loader = yaml.FullLoader,
    )
    
    # Init model - let gridsearch handle parameter init & assignment
    model = FixedWidthModel
    
    # Define gridsearch config
    model_config = {
        "save_fpath":f"{args.save_fpath}/nn_{args.analysis_id}",
        "save_model":False,
        "save_params":True,
        "n_searches":args.n_searches,
        "random_search":True,
        }
    
    # perform grid search
    m = gridsearch_nn(model, param_grid = param_grid, **model_config)
    m.fit(loaders)