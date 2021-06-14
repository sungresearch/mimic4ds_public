"""
Train models under the domain generalization framework
- leave one domain out (left out domain: 2017 - 2019)
"""
import pandas as pd
import numpy as np

import argparse
import os
import pickle
import yaml
import torch

from utils_prediction.utils import str2bool 
from utils_prediction.dataloader.mimic4 import dataloader
from utils_prediction.nn.group_fairness import group_regularized_model
from utils_prediction.nn.robustness import group_robust_model
from utils_prediction.nn.models import FixedWidthModel
from utils_prediction.model_evaluation import StandardEvaluator

## Init parser
parser = argparse.ArgumentParser(
    description = "Train DNN using ERM and Domain Generalization Methods"
)

## args - required without defaults
# general
parser.add_argument(
    "--analysis_id", 
    type = str, 
    required = True, 
    help="Task"
)

## args with defaults
# dataset related
parser.add_argument(
    "--artifacts_fpath",
    type = str,
    default = "/hpf/projects/lsung/projects/public/mimic4ds_public/mimic4ds/Experiments/domain_gen/artifacts",
    help = "path where datasets folder is stored"
)

parser.add_argument(
    "--datasets_ftype",
    type = str,
    default = "parquet",
    help = "file type [parquet (default) or feather] of datasets"
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
    default = "/hpf/projects/lsung/projects/public/mimic4ds_public/mimic4ds/Experiments/domain_gen/hyperparams",
    help = "path where hyperparameters for models and learning algorithms are stored"
)

## group regularized model parameters
parser.add_argument(
    "--train_method",
    default = 'al_layer',
    help = "training method ['erm','irm','al_layer','dro','coral']"
)

## addtional group params
parser.add_argument(
    "--lambd",
    type = float,
    default = -1.0,
    help = "Optional lambda hparam for lambda sweeping. If lambd>0, overrides lambd selected by gridsearch. "
)

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
    help = "seed"
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
    fname = 'preprocessed_dense'
    data = dataloader(**dataloader_config).load_datasets(fname)
    
    # get torch dataloader
    loaders = data.to_torch(
        group_var_name='group_var'
    )
    del data
    return loaders


def get_model(args):
    """
    return model based with training method
    available methods:
        - ERM: standard model (fixed-width NN)
        - AL: standard model with adversarial learning (fixed-width NN with GroupAdversarialModel)
    """
    if args['train_method'] == 'erm':
        fname = f"nn_{args['analysis_id']}"
        fpath = f"{args['hyperparams_fpath']}/models"
    else:
        fname = f"nn_{args['analysis_id']}_{args['train_method']}"
        fpath = f"{args['hyperparams_fpath']}/algo"
    
    # load nn hyper params
    f = open(f"{fpath}/{fname}/model_params.yml",'r')
    param = yaml.load(f,Loader=yaml.FullLoader)
    
    if args['lambd']>0:
        if args['train_method'] == 'dro':
            param['lr_lambda'] = args['lambd']
        else:
            param['lambda_group_regularization'] = args['lambd']
    
    if args['train_method']=='erm':
        # init model
        m = FixedWidthModel(**param)
    elif args['train_method']=='irm':
        # init model
        model_class = group_regularized_model(model_type='group_irm')
        m = model_class(**param)
    elif 'al' in args['train_method'] and 'coral' not in args['train_method']:
        model_class = group_regularized_model(model_type='adversarial')
        m = model_class(**param)   
    elif args['train_method']=='dro':
        model_class = group_robust_model(model_type='loss')
        m = model_class(**param)
    elif args['train_method']=='dro_irm':
        model_class = group_robust_model(model_type='IRM_penalty_proxy')
        m = model_class(**param)
    elif args['train_method']=='coral' or args['train_method']=='coral_0':
        model_class = group_regularized_model(model_type='group_coral')
        m = model_class(**param)
    return m

def save_model(args, m):
    """
    save weights to location defined by training method & parameters
    """
    
    print('saving model weights')
    
    fpath = f"{args['artifacts_fpath']}/analysis_id={args['analysis_id']}/models/"
    if args['lambd']>0:
        fpath += f"nn_{args['train_method']}_lambda_{args['lambd']}_{args['i_iter']}"
    else:
        fpath += f"nn_{args['train_method']}_{args['i_iter']}"
    
    if not os.path.exists(fpath):
        os.makedirs(fpath)
        
    fpath += '/model_weights'
    
    m.save_weights(f"{fpath}")
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
    print(f"Train model with {args['train_method']}")
    print(f"task:{args['analysis_id']}")
    if args['lambd']>0: print(f"Override lambda group regularization with {args['lambd']}")
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