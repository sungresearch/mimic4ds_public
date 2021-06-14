"""
Train model using gridsearch
"""

import argparse, os, pickle, yaml

from utils_prediction.utils import str2bool 
from utils_prediction.dataloader.mimic4 import dataloader
from utils_prediction.model_selection import gridsearch, gridsearch_nn
from utils_prediction.nn.models import FixedWidthModel

from sklearn.linear_model import LogisticRegression as lr
from sklearn.ensemble import RandomForestClassifier as rf
from xgboost import XGBClassifier as xgb

import torch
import numpy as np

## Init parser
parser = argparse.ArgumentParser(
    description = "Train model via gridsearch"
    )

## args - required without defaults
# general
parser.add_argument("--analysis_id", type = str, required = True, help="task: mortality or longlos")

# datasets related
parser.add_argument(
    "--group", 
    type = str, 
    required = True, 
    help="group to select: all, 2008 - 2010, 2011 - 2013, 2014 - 2016, 2017 - 2019")

# gridsearch related
parser.add_argument(
    "--model",
    type = str,
    required = True,
    help = 'model to train [nn, lr, rf, xgb]'
    )

## args with defaults
# dataset related
parser.add_argument(
    "--datasets_fpath",
    type = str,
    default = "/hpf/projects/lsung/projects/public/mimic4ds_public/mimic4ds/Experiments/baseline/artifacts",
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

# preprocessor related
parser.add_argument(
    "--preprocessors_fpath",
    type = str,
    default = "/hpf/projects/lsung/projects/public/mimic4ds_public/mimic4ds/Experiments/baseline/artifacts/",
    help = "path where preprocessors are stored"
    )

# gridsearch related
parser.add_argument(
    "--grids_path",
    type = str,
    default = "/hpf/projects/lsung/projects/public/mimic4ds_public/mimic4ds/Experiments/baseline/hyperparams",
    help = 'path where hyperparameter grids are stored'
    )

parser.add_argument(
    "--save_fpath",
    type = str,
    default = "/hpf/projects/lsung/projects/public/mimic4ds_public/mimic4ds/Experiments/baseline/hyperparams/models",
    help = 'path where hyperparameter grids are stored'
    )

parser.add_argument(
    "--n_jobs",
    type = int,
    default = 4,
    help = "number of parallel searches [default 4]"
    )

parser.add_argument(
    "--n_searches",
    type = int,
    default = 100,
    help = "number of total random searches [default 50]"
    )

parser.add_argument(
    "--save_model",
    type = str2bool,
    default = 'true',
    help = "whether to save trained model"
    )

parser.add_argument(
    "--save_tag",
    type = str,
    default = "baseline",
    help = "tag to add to model name [default: baseline] - models are saved as 'model_group_tag'"
    )

parser.add_argument(
    "--seed",
    type = int,
    default = 44,
    help = "Seed for deterministic training"
    )

## Specify models
models = {
    "nn":FixedWidthModel,
    "lr":lr,
    "rf":rf,
    "xgb":xgb
    }

### run
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
    print(f"loading datasets from {args.datasets_fpath}/analysis_id={args.analysis_id}/datasets/{args.group}/")
    data = dataloader(**dataloader_config).load_datasets(fname=args.group)
    
    ## Preprocessor
    fname = args.group
    fpath = f"{args.preprocessors_fpath}/analysis_id={args.analysis_id}/preprocessors"
    f = open(f"{fpath}/{fname}.pkl","rb")
    print(f"loading preprocessors from {fpath}")
    pipe = pickle.load(f)
    
    # preprocess all datasets**
    # note that although this is a required step, model training via grid search is actually not 
    # using the test sets. The reason why we're transforming the test set here is that
    # to_torch() for nn training requires that all datasets are preprocessed. 
    # X_train and X_val are already preprocessed
    data.X_train = pipe.transform(data.X_train)
    data.X_val = pipe.transform(data.X_val)
    data.X_test = pipe.transform(data.X_test)
    
    ## Gridsearch
    # load grid
    fname = f"{args.grids_path}/{args.model}.yml"
    print(f"loading hyper-parameter grid from {fname}")
    param_grid = yaml.load(open(fname,'r'), Loader = yaml.FullLoader)
    
    # Get model
    model = models[args.model]
    
    # Define model config
    gs_config = {
        "save_fpath":f"{args.save_fpath}/{args.model}_{args.analysis_id}_{args.group}_{args.save_tag}",
        "save_model":False,
        "save_params":True,
        "n_searches":args.n_searches,
        "random_search":True,
    }
    
    
    # perform grid search
    if args.model == 'nn':
        loaders = data.to_torch()
        m = gridsearch_nn(model, param_grid = param_grid, **gs_config)
        m.fit(loaders)
    else:
        data = data.to_sparse()
        m = gridsearch(model, param_grid = param_grid, **gs_config)
        m.fit(data.train['X'], data.train['y'], data.val['X'], data.val['y'])