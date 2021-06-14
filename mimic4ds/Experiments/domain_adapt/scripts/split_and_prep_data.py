import argparse
import os
import pickle

import pandas as pd
import numpy as np

from copy import deepcopy
from imblearn.pipeline import Pipeline

from utils_prediction.utils import str2bool
from utils_prediction.dataloader.mimic4 import dataloader
from utils_prediction.preprocessor import (
    fill_missing, 
    prune_features, 
    binary_discretizer, 
    discretizer, 
    one_hot_encoder
)


## Init parser
parser = argparse.ArgumentParser(
    description = "Loads, splits, preprocesses, and saves features for domain adaptation experiment"
)

## args - requried without defaults
parser.add_argument(
    "--analysis_id", 
    type = str, 
    required = True, 
    help="task: mortality or longlos"
)

## args - optional with defaults
parser.add_argument(
    "--n_ood_list",
    nargs="+",
    default = [100,500,1000,1500],
    help = "number of ood samples to train domain adaptation algorithms"
)

parser.add_argument(
    "--baseline_artifacts_fpath",
    type = str,
    default = "/hpf/projects/lsung/projects/public/mimic4ds_public/mimic4ds/Experiments/baseline/artifacts",
    help = "path where artifacts from baseline experiment are stored"
)

parser.add_argument(
    "--domain_gen_artifacts_fpath",
    type = str,
    default = "/hpf/projects/lsung/projects/public/mimic4ds_public/mimic4ds/Experiments/domain_gen/artifacts",
    help = "path where artifacts from domain_gen experiment are stored"
)

parser.add_argument(
    "--domain_adapt_artifacts_fpath",
    type = str,
    default = "/hpf/projects/lsung/projects/public/mimic4ds_public/mimic4ds/Experiments/domain_adapt/artifacts",
    help = "path where artifacts from domain_adapt experiment are stored"
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

parser.add_argument(
    "--seed", 
    type = int, 
    default = 44
)

############################## helper funcs #########################
def get_domain_gen_data(args):
    """
    Get raw data (after train-val-test split) from domain_gen/erm folder
    """
    
    # dataloader configs
    dataloader_config = {
        'analysis_id':args['analysis_id'],
        'datasets_fpath':args['domain_gen_artifacts_fpath'],
        'datasets_ftype':args['datasets_ftype'],
        'verbose':args['loader_verbose']
    }
    
    # load data
    fname = "raw_split"
    data = dataloader(**dataloader_config).load_datasets(fname)
    return data

def get_baseline_2017_data(args):
    """Get 2017 - 2019 features from baseline experiment.
    Extract features of 2017 - 2019 subject ids from baseline 
    features not in the test set of domain_gen/erm features.
    """
    # dataloader configs
    dataloader_config = {
        'analysis_id':args['analysis_id'],
        'datasets_fpath':args['baseline_artifacts_fpath'],
        'datasets_ftype':args['datasets_ftype'],
        'verbose':args['loader_verbose']
    }
    
    # load data
    fname = "2017 - 2019"
    data = dataloader(**dataloader_config).load_datasets(fname)
    
    X = data.X_train.sample(n=args['n_ood'], random_state = args['seed'])
    X['group_var']='2017 - 2019'
    # labels are taken as well. They won't be used to train
    # domain adaptation models (i.e., the weights of these
    # ood test samples will be set to 0 during training). 
    # The labels are needed for torch dataloader.
    y = data.y_train[X.index.tolist()]
    ids = data.ids_train[X.index.tolist()]
    X = X.reset_index(drop=True)
    
    return (X,y,ids)

############################### run script ##########################
if __name__ == "__main__":
    
    args = vars(parser.parse_args())
    
    # get data
    print('loading domain gen / erm data...')
    domain_gen_data = get_domain_gen_data(args)
    
    # get ood data, join with domain_gen_data
    for n_ood in args['n_ood_list']:
        
        data = deepcopy(domain_gen_data)
        data.config_dict['datasets_fpath'] = args['domain_adapt_artifacts_fpath']
        
        args['n_ood'] = n_ood
        
        print(f"Getting {n_ood} OOD samples from target domain (2017 - 2019)")
        X,y,ids = get_baseline_2017_data(args)
        
        print("Joining OOD samples w/ training data")
        data.X_train = pd.concat((data.X_train, X), axis=0, ignore_index=True)
        data.y_train = np.concatenate((data.y_train, y))
        data.ids_train = np.concatenate((data.ids_train, ids))
        
        print("Saving raw dataset")
        data.save_datasets(f"raw_split_{n_ood}")
        
        # Load Domain_Gen / ERM Preprocessors 
        print('loading Domain Gen / ERM preprocesors')
        
        fpath = os.path.join(
            args['domain_gen_artifacts_fpath'],
            f"analysis_id={args['analysis_id']}",
            "preprocessors"
        )
        
        fname = 'preprocessors'
        f = open(f"{fpath}/{fname}.pkl","rb")
        m = pickle.load(f)
        
        print("Preprocessing data")
        data.X_train = m.transform(data.X_train)
        data.X_val = m.transform(data.X_val)
        data.X_test = m.transform(data.X_test)
        
        print("Saving preprocessed dataset")
        data.save_datasets(f"preprocessed_dense_{n_ood}")
