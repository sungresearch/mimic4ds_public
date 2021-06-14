import pandas as pd
import numpy as np

import os
import argparse
import yaml
import pickle

from utils_prediction.model_evaluation import StandardEvaluator
from utils_prediction.dataloader.mimic4 import *
from utils_prediction.utils import str2bool

parser = argparse.ArgumentParser(description = "Compare model performance")

## args - requried without defaults
# General
parser.add_argument(
    "--analysis_id", 
    type = str, 
    required = True, 
    help="task: mortality, longlos, invasivevent, sepsis"
)

parser.add_argument(
    "--train_group_1", 
    type = str, 
    required = True, 
    help="group the model/preprocessors were trained on \
    [all, 2008 - 2010, 2011 - 2013, 2014 - 2016, 2017 - 2019]"
)

parser.add_argument(
    "--train_group_2", 
    type = str, 
    required = True, 
    help="group the model/preprocessors were trained on \
    [all, 2008 - 2010, 2011 - 2013, 2014 - 2016, 2017 - 2019]"
)

parser.add_argument(
    "--eval_group",
    type = str,
    required = True,
    help="group to evaluate the model/preprocessors on \
    [all, 2008 - 2010, 2011 - 2013, 2014 - 2016, 2017 - 2019]"
)

## args - with defaults
# general
parser.add_argument(
    "--artifacts_fpath",
    type = str,
    default = "/hpf/projects/lsung/projects/public/mimic4ds_public/mimic4ds/Experiments/baseline/artifacts",
    help = "general path where datasets, preprocessors, \
    models, and results are stored"
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
    "--evaluation_method",
    type = str,
    default = "avg",
    help = "Whether to bootstrap evaluate avg, \
    ensemble, or best model performance"
)

# ---------------------------------------------
# helper funcs
# ---------------------------------------------
def compare(args, df_base, df_test):
    """bootstrap compare models to obtain difference between means 
    at each iteration
    
    df_base: results of models trained using data from 08-10
    df_test: results of models trained using data from subsequent years
    """
    df_results = pd.DataFrame()
    evaluator = StandardEvaluator()
    
    # bootstrap
    for i in range(args['n_boot']):
        
        i_df_base = df_base.groupby(["phase", "iter"]).sample(frac=1, replace=True)
        i_df_test = df_test.loc[i_df_base.index,:]
        
        i_metrics_base = evaluator.evaluate(
            i_df_base,
            strata_vars = ['phase', 'iter']
        )
        
        i_metrics_test = evaluator.evaluate(
            i_df_test,
            strata_vars = ['phase', 'iter']
        )
        
        i_mean_base = i_metrics_base.groupby(
            ["phase","metric"]
        )['performance'].mean().reset_index()
        
        i_mean_test = i_metrics_test.groupby(
            ["phase","metric"]
        )['performance'].mean().reset_index()
        
        i_diff = i_mean_base.copy()
        i_diff['performance_test'] = i_mean_test['performance']
        i_diff['performance_diff'] = i_mean_base['performance'] - i_mean_test['performance']
        
        i_diff.rename(
            columns={'performance':'performance_base'}, 
            inplace=True
        )
        
        i_diff['boot_iter'] = i+1
        
        df_results = pd.concat((df_results, i_diff),axis=0)
    
    return df_results

def compare_ensemble(args,df_base, df_test):
    """bootstrap compare average model outputs 
    to obtain difference in ensemble performance
    """
    df_results = pd.DataFrame()
    evaluator = StandardEvaluator()
    
    # bootstrap
    for i in range(args['n_boot']):
        
        i_df_base = df_base.groupby(["phase", "iter"]).sample(frac=1, replace=True)
        i_df_test = df_test.loc[i_df_base.index,:]
        
        i_df_base = i_df_base.groupby(["phase","ids"]).mean().reset_index()
        i_df_test = i_df_test.groupby(["phase","ids"]).mean().reset_index()
        
        i_metrics_base = evaluator.evaluate(
            i_df_base, 
            strata_vars = ['phase'],
        )
        
        i_metrics_test = evaluator.evaluate(
            i_df_test, 
            strata_vars = ['phase'],
        )
        
        i_diff = i_metrics_base.copy()
        i_diff['performance_test'] = i_metrics_test['performance']
        i_diff['performance_diff'] = i_metrics_base['performance'] - i_metrics_test['performance']
        
        i_diff.rename(
            columns={'performance':'performance_base'}, 
            inplace=True
        )
        
        i_diff['boot_iter'] = i+1
        df_results = pd.concat((df_results, i_diff),axis=0)
        
    return df_results
        
def compare_best(args,df_base, df_test):
    """bootstrap compare best model determined based on 
    validation performance
    """
    df_results = pd.DataFrame()
    evaluator = StandardEvaluator()
    
    # get best base model based on validation bce loss
    #breakpoint()
    
    performance_base = evaluator.evaluate(
        df_base.query("phase=='val'"), 
        strata_vars=['iter']
    )
    
    idx_best_model_base = performance_base.groupby(
        'metric'
    )['performance'].idxmin()['loss_bce']
    
    best_iter_base = performance_base.loc[idx_best_model_base,'iter']
    df_base = df_base.query("iter==@best_iter_base")
    
    # get best test model based on validation bce loss
    performance_test = evaluator.evaluate(
        df_test.query("phase=='val'"), 
        strata_vars=['iter']
    )
    
    idx_best_model_test = performance_test.groupby(
        'metric'
    )['performance'].idxmin()['loss_bce']
    
    best_iter_test = performance_test.loc[idx_best_model_test,'iter']
    df_test = df_test.query("iter==@best_iter_test")
    
    # bootstrap
    for i in range(args['n_boot']):
        i_df_base = df_base.groupby('phase').sample(frac=1, replace=True)
        i_df_test = df_test.query("ids==@i_df_base['ids'].tolist()")
        
        i_metrics_base = evaluator.evaluate(
            i_df_base,
            strata_vars=['phase']
        )
        
        i_metrics_test = evaluator.evaluate(
            i_df_test,
            strata_vars=['phase']
        )
        
        i_diff = i_metrics_base.copy()
        i_diff['performance_test'] = i_metrics_test['performance']
        i_diff['performance_diff'] = i_metrics_base['performance'] - i_metrics_test['performance']
        
        i_diff.rename(
            columns={'performance':'performance_base'}, 
            inplace=True
        )
        
        i_diff['boot_iter'] = i+1
        df_results = pd.concat((df_results, i_diff), axis=0)
    
    return df_results

# ---------------------------------------------
# run
# ---------------------------------------------
if __name__ == "__main__":
    
    args = vars(parser.parse_args())
    np.random.seed(args['seed'])
    
    ## print
    print('#####'*10)
    print(f"Compare {args['train_group_1']} models (base) w/ {args['train_group_2']} models (test)")
    print(f"Evaluating on {args['eval_group']} data")
    print(f"task:{args['analysis_id']}")
    print(f"n_boot: {args['n_boot']}")
    print('#####'*10)
    
    # get model outputs 
    fpath = os.path.join(
        args['artifacts_fpath'],
        f"analysis_id={args['analysis_id']}",
        "results",
        "pred_probs",
    )
    
    fname_base = "_".join([
        "nn",
        args['train_group_1'],
        args['eval_group']
    ])
    
    fname_test = "_".join([
        "nn",
        args['train_group_2'],
        args['eval_group']
    ])
    
    print("Loading model outputs...")
    df_base = pd.read_csv(f"{fpath}/{fname_base}.csv")
    df_test = pd.read_csv(f"{fpath}/{fname_test}.csv")
    
    ids_base = df_base.query("phase=='test'")['ids'].values
    ids_test = df_test.query("phase=='test'")['ids'].values
    
    # Check that row ids are equal across dfs to be compared
    assert(np.sum(ids_base==ids_test)==len(ids_base)), (
        'Row IDs between results need to be equal in order to compare')
    
    # bootstrap evaluate models
    print("Comparing model performance...")
    
    if args['evaluation_method'] == 'avg':
        df_results = compare(args, df_base, df_test)
        
    elif args['evaluation_method'] == 'ensemble':
        df_results = compare_ensemble(args, df_base, df_test)
        
    elif args['evaluation_method'] == 'best':
        df_results = compare_best(args, df_base, df_test)
    
    # add additional info to df
    df_results['train_group_1'] = args['train_group_1']
    df_results['train_group_2'] = args['train_group_2']
    df_results['eval_group'] = args['eval_group']
    df_results['analysis_id'] = args['analysis_id']
    
    # save evaluation
    print("saving evaluation...")
    fpath = os.path.join(
        args['artifacts_fpath'],
        f"analysis_id={args['analysis_id']}",
        'results',
        'compare_models',
    )
    os.makedirs(fpath, exist_ok = True)
    
    fname = '_'.join(
        filter(None, [
            'nn',
            args['train_group_1'],
            args['train_group_2'],
            args['eval_group'],
            args['evaluation_method'],
        ])
    )
    
    df_results.to_csv(f"{fpath}/{fname}.csv")