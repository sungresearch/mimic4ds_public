import pandas as pd
import numpy as np

import os
import argparse
import yaml
import pickle

from utils_prediction.model_evaluation import StandardEvaluator
from utils_prediction.dataloader.mimic4 import *
from utils_prediction.nn.models import FixedWidthModel
from utils_prediction.utils import str2bool

parser = argparse.ArgumentParser(description = "Get model predictions")

## args - requried without defaults
# General
parser.add_argument(
    "--analysis_id", 
    type = str, 
    required = True, 
    help="task: mortality, longlos, invasivevent, sepsis"
)

parser.add_argument(
    "--train_method", 
    type = str, 
    required = True, 
    help="Method used to train models \
    ['erm','irm','dro','al_layer','coral']"
)


## args - with defaults
# general
parser.add_argument(
    "--demographic_category",
    type = str,
    default = 'gender',
    help = "Which demographic category to stratify subjects"
)

parser.add_argument(
    "--artifacts_fpath",
    type = str,
    default = "/hpf/projects/lsung/projects/public/mimic4ds_public/mimic4ds/Experiments/domain_adapt/artifacts",
    help = "general path where datasets, preprocessors, \
    models, and results are stored"
)

# datasets related
parser.add_argument(
    "--n_ood",
    type = int,
    default = 100,
    help = "number of OOD samples used to train models"
)

parser.add_argument(
    "--datasets_ftype",
    type = str,
    default = "parquet",
    help = "file type [parquet (default) or feather] of \
    the datasets to be saved"
)

parser.add_argument(
    "--loader_verbose",
    type = str2bool,
    default = "false",
    help = 'verbosity for dataloader [default: False]'
)

# bootstrap related
parser.add_argument(
    "--n_boot",
    type = int,
    default = 10000,
    help = "Number of bootstrap iterations"
)

parser.add_argument(
    "--seed",
    type = int,
    default = 44,
    help = "Seed for numpy random"
)

parser.add_argument(
    "--group_train_domains",
    type = str2bool,
    default = 'true',
    help = 'wheather to group all training domains as the same.'
)

parser.add_argument(
    "--eval",
    type = str2bool,
    default = 'true',
    help = "whether to run evaluation of outputs"
)
# ---------------------------------------------
# helper funcs
# ---------------------------------------------
def get_data(args):
    """
    Grab demographic info
    """
    ## datasets
    dataloader_config = {
        "analysis_id":args['analysis_id'],
        "datasets_fpath":args['artifacts_fpath'],
        "datasets_ftype":args['datasets_ftype'],
        "verbose":args['loader_verbose']
    }
        
    data = dataloader(
        **dataloader_config
    ).load_datasets(fname=f"raw_split_{args['n_ood']}")
    
    # add ids to dataframes
    data.X_val['ids'] = data.ids_val
    data.X_test['ids'] = data.ids_test
    
    # join demographics info from validation & test sets
    df = pd.concat((
        data.X_val[[
            'ids','insurance','marital_status','ethnicity','language','age_measurement','gender'
        ]], 
        data.X_test[[
            'ids','insurance','marital_status','ethnicity','language','age_measurement','gender'
        ]], 
    ), axis=0)
    
    del(data)
    
    return df
    
def evaluate(args, df):
    """bootstrap evaluate models to obtain mean & std across models
    at each iteration
    """
    df_results = pd.DataFrame()
    evaluator = StandardEvaluator()
    
    # bootstrap
    for i in range(args['n_boot']):
        i_df = df.groupby(
            ["phase", "group", "iter", args['demographic_category']]
        ).sample(frac=1, replace=True)
        
        i_metrics = evaluator.evaluate(
            i_df,
            strata_vars = ['phase', 'group','iter',args['demographic_category']]
        )
        
        i_mean = i_metrics.groupby(["phase","group",args['demographic_category'],"metric"])['performance'].agg(
            ['mean','std']
        ).reset_index()
        
        i_mean['boot_iter'] = i+1
        df_results = pd.concat((df_results, i_mean),axis=0)
    
    return df_results

# ---------------------------------------------
# run
# ---------------------------------------------
if __name__ == "__main__":
    
    args = vars(parser.parse_args())
    np.random.seed(args['seed'])
    
    ## print
    print('#####'*10)
    print(
        f"{args['demographic_category']}-stratified evaluation of \
        nn trained under {args['train_method']}"
    )
    print(f"task:{args['analysis_id']}")
    print(f"n_boot: {args['n_boot']}")
    print('#####'*10)
    
    ##get pred_probs if files do not already exist
    fname = '_'.join(
        filter(None,[
            'nn',
            args['train_method'],
            str(args['n_ood']),
        ])
    )
    fpath = os.path.join(
        args['artifacts_fpath'],
        f"analysis_id={args['analysis_id']}",
        'results',
        'by_demographic',
        'pred_probs',
        f"{fname}.csv"
    )
    
    if os.path.exists(fpath):
        get_pred_probs = False
        outputs = pd.read_csv(fpath)
    else:
        get_pred_probs = True
    
    if get_pred_probs:
        ## get pred_probs
        fname = '_'.join(
            filter(None,[
                'nn',
                args['train_method'],
                str(args['n_ood']),
            ])
        )
        fpath = os.path.join(
            args['artifacts_fpath'],
            f"analysis_id={args['analysis_id']}",
            'results',
            'pred_probs',
            f"{fname}.csv"
        )

        outputs = pd.read_csv(fpath)

        ## Get demographic information
        print("Getting demographic info...")
        df_demographic_info = get_data(args)

        ## Check that ids are the same
        assert(
            len([
                x for x in outputs['ids'].unique() 
                if x in df_demographic_info['ids'].unique()
            ]) == len(outputs['ids'].unique())
              ), (
        'Mismatch in subject ids!')

        outputs = outputs.merge(
            df_demographic_info,
            how='left',
            left_on='ids',
            right_on='ids'
        )

        # save outputs
        print("saving outputs...")
        fpath = os.path.join(
            args['artifacts_fpath'],
            f"analysis_id={args['analysis_id']}",
            'results',
            'by_demographic',
            'pred_probs'
        )

        os.makedirs(fpath, exist_ok = True)

        fname = '_'.join(
            filter(None,[
                'nn',
                args['train_method'],
                str(args['n_ood']),
            ])
        )

        outputs.to_csv(f"{fpath}/{fname}.csv")
    
    if args['eval']:
        # bootstrap evaluate models
        print("evaluating outputs...")
        
        # group ID domains if specified
        if args['group_train_domains']:
            outputs['group'].replace(
                {
                    0:0,
                    1:0,
                    2:0,
                    3:1
                },
                inplace=True
            )

        # run evaluation
        df_results = evaluate(args, outputs)
        
        # add additional info to df
        df_results['train_method'] = args['train_method']
        df_results['analysis_id'] = args['analysis_id']
        df_results['demographic_category'] = args['demographic_category']
        df_results['n_ood'] = args['n_ood']
        
        # save evaluation
        print("saving evaluation...")
        fpath = os.path.join(
            args['artifacts_fpath'],
            f"analysis_id={args['analysis_id']}",
            'results',
            'by_demographic',
            'evaluate_models'
        )
        os.makedirs(fpath, exist_ok = True)

        fname = '_'.join(
            filter(None, [
                'nn',
                args['train_method'],
                str(args['n_ood']),
                args['demographic_category']
            ])
        )

        df_results.to_csv(f"{fpath}/{fname}.csv")