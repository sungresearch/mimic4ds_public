"""
Split into datasets and fit preprocessors
"""

import argparse, os
import pickle
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
    description = "Loads, splits, preprocesses, and saves features"
    )

## args - requried without defaults
parser.add_argument("--analysis_id", type = str, required = True, help="task: mortality or longlos")

parser.add_argument(
    "--group", 
    type = str, 
    required = True, 
    help="group to select: all, 2008 - 2010, 2011 - 2013, 2014 - 2016, 2017 - 2019")

## args - optional with defaults
parser.add_argument(
    "--features_fpath",
    type = str,
    default = "/hpf/projects/lsung/projects/public/mimic4ds_public/mimic4ds/data",
    help = "path where features are extracted to [default: project-folder/data]"
    )

parser.add_argument(
    "--features_ftype",
    type = str,
    default = "parquet",
    help = "file type [parquet (default) or feather] of the extracted feature table"
    )

parser.add_argument(
    "--datasets_fpath",
    type = str,
    default = "/hpf/projects/lsung/projects/public/mimic4ds_public/mimic4ds/Experiments/baseline/artifacts",
    help = "path to save datasets (post splitting)"
    )

parser.add_argument(
    "--datasets_ftype",
    type = str,
    default = "parquet",
    help = "file type [parquet (default) or feather] of the datasets to be saved"
    )

parser.add_argument(
    "--preprocessors_fpath",
    type = str,
    default = "/hpf/projects/lsung/projects/public/mimic4ds_public/mimic4ds/Experiments/baseline/artifacts",
    help = "path to save preprocessors (after fitting)"
    )

parser.add_argument(
    "--loader_verbose",
    type = str2bool,
    default = "false",
    help = 'verbosity for dataloader [default: False]'
    )

parser.add_argument(
    "--p_splits", 
    nargs = '+', 
    default = [0.7, 0.15, 0.15], 
    type = float, 
    help="Proportions to split data [default: 0.7, 0.15, 0.15]"
    )

parser.add_argument("--seed", type = int, default = 44)
parser.add_argument("--label_col", type = str, default = "label", help="column name for the labels")
parser.add_argument("--id_col", type = str, default = "subject_id", help="column name for subject/stay ids")
parser.add_argument("--debug", type = str2bool, default = 'false', help="if true, a sub-sample of data will be used")

        
### run
if __name__ == "__main__":
    
    args = parser.parse_args()
    
    ## dataloader_config
    dataloader_config = {
        'analysis_id':args.analysis_id,
        'features_fpath':args.features_fpath,
        'features_ftype':args.features_ftype,
        'datasets_fpath':args.datasets_fpath,
        'datasets_ftype':args.datasets_ftype,
        'verbose':args.loader_verbose,
        'label_col':args.label_col,
        'id_col':args.id_col
        }
    
    # time-agnostic ('all') or time-specific (2008 - 2010, 2011 - 2013, 2014 - 2016, 2017 - 2019)
    if args.group != 'all': dataloader_config['group'] = args.group
        
    ## load features
    print(f"loading features from {args.group}...")
    data = dataloader(**dataloader_config).load_features()
    
    if args.debug:
        print("In debug mode")
        data.features = data.sample()
    
    ## split data
    print(f"splitting into datasets with proportions {args.p_splits}...")
    data = data.split(p_splits = args.p_splits, seed = args.seed)
    
    ## Preprocess datasets
    pipe = Pipeline([
        ('fill missings',fill_missing(config={'count':0,'marital_status':'None'})),
        ('prune features',prune_features(special_cols={'count':0})),
        ('discretize counts', binary_discretizer(feature_tags_to_include= ['count'])),
        ('discretize measurements', discretizer(feature_tags_to_include = ['measurement'])),
        ('one hot encode', one_hot_encoder(feature_tags_to_exclude = ['count']))
        ])
    
    # Fit pipeline
    print("Fitting preprocessors on X_train...")
    pipe.fit(data.X_train)
    
    ## Save datasets
    fname = args.group
    fpath = f"{args.datasets_fpath}/analysis_id={args.analysis_id}/datasets/{fname}/"
    
    if not args.debug:
        print(f"saving datasets to {fpath}...")
        data.save_datasets(fname) # dataloader.save_datasets auto saves file to the path above
    
    ## Save preprocessors
    fname = args.group
    fpath = f"{args.preprocessors_fpath}/analysis_id={args.analysis_id}/preprocessors"
    
    if not os.path.exists(fpath):
        os.makedirs(fpath)
    
    if not args.debug:
        print(f"saving preprocessors to {fpath}/{fname}...")
        f = open(f"{fpath}/{fname}.pkl","wb")
        pickle.dump(pipe,f)