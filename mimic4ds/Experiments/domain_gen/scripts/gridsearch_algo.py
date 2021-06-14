"""
Train model using gridsearch
- Take model hparams selected using ERM to find best hparams of learning algos (irm, al)
"""

import argparse
import os
import pickle
import yaml
import torch

import numpy as np

from utils_prediction.dataloader.mimic4 import dataloader
from utils_prediction.model_selection import gridsearch, gridsearch_nn
from utils_prediction.nn.group_fairness import group_regularized_model
from utils_prediction.nn.robustness import group_robust_model
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
parser.add_argument("--analysis_id", type = str, required = True, help="task: mortality or longlos")

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
    default = "/hpf/projects/lsung/projects/public/mimic4ds_public/mimic4ds/Experiments/domain_gen/hyperparams/",
    help = 'Path to file with hyperparameter grid.'
    )

parser.add_argument(
    "--model_hparams",
    type = str,
    default = "/models",
    help = "sub-path to selected model hparams"
    )

parser.add_argument(
    "--training_method",
    type = str,
    default = "al",
    help = "training method [ \
        'erm','irm','al','al_layer','al_layer_0',dro','dro_irm','coral','coral_0' \
        'al_layer_sh', 'al_layer_gradrev' \
    ]"
    )

parser.add_argument(
    "--save_fpath",
    type = str,
    default = "/hpf/projects/lsung/projects/public/mimic4ds_public/mimic4ds/Experiments/domain_gen/hyperparams/algo/"
)

parser.add_argument(
    "--n_searches",
    type = int,
    default = 50,
    help = "number of total random searches [default 50]"
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

def get_grid(args):
    fname = f"{args.grids_path}/models/nn_{args.analysis_id}/model_params.yml"
    print(f"loading selected model hyper-parameter grid from {fname}")
    model_params = yaml.load(open(fname,'r'), Loader = yaml.FullLoader)
    model_params = {key:[value] for key,value in model_params.items()}
    
    fname = f"{args.grids_path}/{args.training_method}_grid.yml"
    algo_grid = yaml.load(open(fname,'r'), Loader = yaml.FullLoader)

    return {**algo_grid, **model_params}

def init_model(args):
    if args.training_method == 'irm':
        return group_regularized_model(model_type='group_irm')
    elif 'al' in args.training_method and 'coral' not in args.training_method:
        return group_regularized_model(model_type='adversarial')
    elif args.training_method == 'dro':
        return group_robust_model(model_type='loss')
    elif args.training_method=='dro_irm':
        return group_robust_model(model_type='IRM_penalty_proxy')
    elif args.training_method=='coral' or args.training_method=='coral_0':
        return group_regularized_model(model_type='group_coral')

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
    param_grid = get_grid(args)
    
    # Init model - let gridsearch handle parameter init & assignment
    model = init_model(args)
    
    # Define gridsearch config
    model_config = {
        "save_fpath":f"{args.save_fpath}/nn_{args.analysis_id}_{args.training_method}",
        "save_model":False,
        "save_params":True,
        "n_searches":args.n_searches,
        "random_search":True,
        }
    
    # perform grid search
    m = gridsearch_nn(model, param_grid = param_grid, **model_config)
    m.fit(loaders)