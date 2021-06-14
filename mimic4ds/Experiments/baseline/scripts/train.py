"""
Train models under the domain generalization framework
- leave one domain out (left out domain: 2017 - 2019)
"""
import pandas as pd
import numpy as np
import argparse, os, pickle, yaml
from copy import deepcopy

from utils_prediction.utils import str2bool 
from utils_prediction.dataloader.mimic4 import dataloader
from utils_prediction.nn.group_fairness import group_regularized_model
from utils_prediction.nn.robustness import group_robust_model
from utils_prediction.nn.models import FixedWidthModel
from utils_prediction.model_evaluation import StandardEvaluator

import torch
import numpy as np

## Init parser
parser = argparse.ArgumentParser(
    description = "Train DNN"
    )

## args - required without defaults
# general
parser.add_argument(
    "--analysis_id", 
    type = str, 
    required = True, 
    help="task: mortality or longlos"
    )

# datasets related
parser.add_argument(
    "--group", 
    type = str, 
    required = True, 
    help="group to select: all, 2008 - 2010, 2011 - 2013, 2014 - 2016, 2017 - 2019")

## args with defaults
# dataset related
# gridsearch related
parser.add_argument(
    "--model",
    default = 'nn',
    type = str,
    help = 'model to train [nn, lr, rf, xgb]'
    )

parser.add_argument(
    "--artifacts_fpath",
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

## model related
parser.add_argument(
    "--hyperparams_fpath",
    type = str,
    default = "/hpf/projects/lsung/projects/public/mimic4ds_public/mimic4ds/Experiments/baseline/hyperparams",
    help = "path where hyperparameters for models and learning algorithms are stored"
    )

## group regularized model parameters
parser.add_argument(
    "--n_iters",
    type = int,
    default = 20,
    help = "how many times to train each model"
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

def get_data(args):
    """
    grab train data and return loaders
    """
    
    # dataloader configs
    dataloader_config = {
        'analysis_id':args['analysis_id'],
        'datasets_fpath':args['artifacts_fpath'],
        'datasets_ftype':args['datasets_ftype'],
        'verbose':args['loader_verbose']
        }
    
    # load data
    data = dataloader(**dataloader_config).load_datasets(fname=args['group'])
    
    ## Preprocessor
    fname = args['group']
    fpath = f"{args['artifacts_fpath']}/analysis_id={args['analysis_id']}/preprocessors"
    f = open(f"{fpath}/{fname}.pkl","rb")
    print(f"loading preprocessors")
    pipe = pickle.load(f)
    
    data.X_train = pipe.transform(data.X_train)
    data.X_val = pipe.transform(data.X_val)
    data.X_test = pipe.transform(data.X_test)
    
    # get torch dataloader
    loaders = data.to_torch()
    del data
    return loaders


def get_model(args):
    """
    return model based with training method
    available methods:
        - ERM: standard model (fixed-width NN)
        - AL: standard model with adversarial learning (fixed-width NN with GroupAdversarialModel)
    """
    fpath = f"{args['hyperparams_fpath']}/models"
    fname = f"{args['model']}_{args['analysis_id']}_{args['group']}_baseline"
    
    # load nn hyper params
    f = open(f"{fpath}/{fname}/model_params.yml",'r')
    param = yaml.load(f,Loader=yaml.FullLoader)
    m = FixedWidthModel(**param)
    return m

def save_model(args, m):
    """
    save weights to location defined by training method & parameters
    """
    
    print('saving model weights')
    
    fname = '_'.join([
        args['model'],
        args['group'],
        str(args['i_iter'])
    ])
    
    fpath = os.path.join(
        args['artifacts_fpath'],
        f"analysis_id={args['analysis_id']}",
        "models",
        fname
    )
    
    os.makedirs(fpath,exist_ok=True)
    
    m.save_weights(f"{fpath}/model_weights")
    return fpath
    

#-------------------------------------------------------------------
# run
#-------------------------------------------------------------------
if __name__ == "__main__":
    
    args = vars(parser.parse_args())
    
    # set seed
    torch.manual_seed(args['seed'])
    np.random.seed(args['seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
        
    ## header
    print('#####'*10)
    print(f"Training {args['n_iters']} {args['model']} models")
    print(f"task:{args['analysis_id']}")
    print('#####'*10)
    
    # get data
    print("loading data...")
    loaders = get_data(args)
    
    # get model
    for i_iter in range(args['n_iters']):
        args['i_iter'] = i_iter
        m = get_model(args)

        # train model
        print('training model...')
        m.train(loaders, phases=['train','val'])

        # save trained model & get weights path
        print('saving weights...')
        _ = save_model(args,m)