import argparse
import os
import pickle
import pdb

import pandas as pd
import numpy as np

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
parser.add_argument(
    "--analysis_id", 
    type = str, 
    required = True, 
    help="task [mortality, longlos, sepsis, invasivevent]"
)

## args - optional with defaults
# data related
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
    default = "/hpf/projects/lsung/projects/public/mimic4ds_public/mimic4ds/Experiments/domain_gen/artifacts",
    help = "path to save datasets (post splitting)"
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

parser.add_argument(
    "--label_col", 
    type = str, 
    default = "label", 
    help="column name for the labels"
)

parser.add_argument(
    "--id_col", 
    type = str, 
    default = "subject_id", 
    help="column name for subject/stay ids"
)

parser.add_argument(
    "--baseline_artifacts_fpath",
    type = str,
    default = "/hpf/projects/lsung/projects/public/mimic4ds_public/mimic4ds/Experiments/baseline/artifacts",
    help = "baseline artifacts contain datasets from which we obtain test ids"
)


#-------------------------------------------------------------------
# helper functions
#-------------------------------------------------------------------
def get_test_ids(args):
    """Obtain test ids from baseline data
    """
    yeargroups = [
        "2008 - 2010",
        "2011 - 2013", 
        "2014 - 2016", 
        "2017 - 2019"
    ]
    
    test_ids = {}
    
    # Extract test ids from each year group
    for yeargroup in yeargroups:
        
        dataloader_config = {
            'analysis_id':args.analysis_id,
            'datasets_fpath':args.baseline_artifacts_fpath,
            'datasets_ftype':args.datasets_ftype,
            'verbose':args.loader_verbose,
        }
        
        data = dataloader(**dataloader_config).load_datasets(yeargroup)
        test_ids[yeargroup] = data.ids_test.tolist()
        del(data)
    
    return test_ids
    
    
def split_data(args, data, test_ids):
    """splits features into train,val,test sets and ensures that 
    the same subjects are in the test set as baseline experiment. 
    
    Split: 
        2008 - 2010: 85% [train] - 15% [test]
        2011 - 2013: 85% [train] - 15% [test]
        2014 - 2016: 45% [train] - 35% [val] - 15% [test]
        2017 - 2019: 15% [test]
    
    Input: dataloader class contains features
    
    output: dataloader class with attributes for downstream processing
        X_train, y_train, ids_train
        X_val, y_val, ids_val
        X_test, y_test, ids_test
    """
    
    # test set
    test = pd.concat((
        data.features.query("subject_id==@ids")
        for _, ids in test_ids.items()
    ))
    
    # train set
    all_test_ids = [
        x for _, ids in test_ids.items()
        for x in ids
    ]
    
    train = data.features.query("\
        subject_id!=@all_test_ids and \
        group_var!='2017 - 2019'\
    ")
    
    # val set
    ids = train.query("\
        group_var=='2014 - 2016'\
    ")['subject_id'].values.tolist()
    
    np.random.shuffle(ids)
    ids_val = ids[:int(0.35/0.85*len(ids))] # 35% of 2014 - 2016
    
    val = train.query("subject_id==@ids_val")
    train = train.drop(index=val.index)
    
    # remove features attribtue from data
    delattr(data,'features')
    
    # assign attributes to data
    setattr(data, 'y_train', train.pop('label').values)
    setattr(data, 'ids_train', train.pop('subject_id').values)
    setattr(data, 'X_train', train.reset_index(drop=True))
    setattr(data, 'y_val', val.pop('label').values)
    setattr(data, 'ids_val', val.pop('subject_id').values)
    setattr(data, 'X_val', val.reset_index(drop=True))
    setattr(data, 'y_test', test.pop('label').values)
    setattr(data, 'ids_test', test.pop('subject_id').values)
    setattr(data, 'X_test', test.reset_index(drop=True))
    
    return data

#-------------------------------------------------------------------
# run
#-------------------------------------------------------------------
if __name__ == "__main__":
    
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    
    # get test subject IDs
    print("getting test ids from baseline experiment...")
    test_ids = get_test_ids(args)
    
    # get features
    dataloader_config = {
        'analysis_id':args.analysis_id,
        'features_fpath':args.features_fpath,
        'features_ftype':args.features_ftype,
        'datasets_fpath':args.datasets_fpath,
        'datasets_ftype':args.datasets_ftype,
        'verbose':args.loader_verbose,
        'label_col':args.label_col,
        'id_col':args.id_col,
        }
        
    print(f"loading features...")
    data = dataloader(**dataloader_config).load_features()
    
    ## split data and use the same test IDs as baseline
    print(f"splitting into train, val, and test sets")
    data = split_data(args, data, test_ids)
    
    ## Preprocess datasets
    pipe = Pipeline([
        ('fill missings',fill_missing(config={'count':0,'marital_status':'None'})),
        ('prune features',prune_features(special_cols={'count':0})),
        ('discretize counts', binary_discretizer(feature_tags_to_include= ['count'])),
        ('discretize measurements', discretizer(feature_tags_to_include = ['measurement'])),
        ('one hot encode', one_hot_encoder(feature_tags_to_exclude = ['count','group_var']))
    ])
    
    # Fit pipeline
    print("Fitting preprocessors on X_train...")
    pipe.fit(data.X_train)
    
    ## Save datasets
    fname = 'raw_split'
    
    fpath = os.path.join(
        args.datasets_fpath,
        f"analysis_id={args.analysis_id}",
        'datasets',
        fname
    )
    
    os.makedirs(fpath, exist_ok=True)
    
    print(f"saving raw, non-preprocessed datasets to {fpath}...")
    # dataloader saves file to the path above because it
    # was specified as the dataset path in the dataloader 
    # config 
    data.save_datasets(fname) 
    
    ## Save preprocessors
    fname = 'preprocessors'
    
    fpath = os.path.join(
        args.datasets_fpath,
        f"analysis_id={args.analysis_id}",
        "preprocessors"
    )
    
    os.makedirs(fpath, exist_ok=True)

    print(f"saving preprocessors to {fpath}/{fname}...")
    f = open(f"{fpath}/{fname}.pkl","wb")
    pickle.dump(pipe,f)
        
    ## preprocess datasets and save
    data.X_train = pipe.transform(data.X_train)
    data.X_val = pipe.transform(data.X_val)
    data.X_test = pipe.transform(data.X_test)
    
    fname = 'preprocessed_dense'
    
    fpath = os.path.join(
        args.datasets_fpath,
        f"analysis_id={args.analysis_id}",
        'datasets',
        fname
    )

    print(f"saving preprocessed datasets to {fpath}...")
    
    # dataloader saves file to the path above because it
    # was specified as the dataset path in the dataloader 
    # config 
    data.save_datasets(fname)