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
    "--train_group", 
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
    "--analysis_tag",
    type = str,
    default = "baseline",
    help = "tag that was added to model name \
    [default:baseline] - models were saved as 'model_group_tag'"
)

parser.add_argument(
    "--artifacts_fpath",
    type = str,
    default = "/hpf/projects/lsung/projects/public/mimic4ds_public/mimic4ds/Experiments/baseline/artifacts",
    help = "general path where datasets, preprocessors, \
    models, and results are stored"
)

# datasets related
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

# preprocessor related
parser.add_argument(
    "--preprocessors_fpath",
    type = str,
    default = "preprocessors",
    help = "path where preprocessors are artifacts/analysis-id/"
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

# Evaluation related
parser.add_argument(
    "--eval",
    type = str2bool,
    default = "true",
    help = "Whether to evaluate model outputs"
)


#parser.add_argument(
#    "--threshold_quantile",
#    type = float,
#    default = 0.05,
#    help = "cut-off for probability threshold"
#)

# ---------------------------------------------
# helper funcs
# ---------------------------------------------
def get_data(args):
    """
    Load both args['eval_group'] and args['train_group']
    Preprocess data using preprocessors trained on args['train_group']
    The validation set from args['train_group'] will be used to 
    find "best" model and also the threshold to compute threshold-based
    metrics
    """
    ## datasets
    dataloader_config = {
        "analysis_id":args['analysis_id'],
        "datasets_fpath":args['artifacts_fpath'],
        "datasets_ftype":args['datasets_ftype'],
        "verbose":args['loader_verbose']
    }
        
    data_train = dataloader(
        **dataloader_config
    ).load_datasets(fname=args['train_group'])
    
    data_eval = dataloader(
        **dataloader_config
    ).load_datasets(fname=args['eval_group'])
    
    # join train & test datasets
    data_train.X_test = data_eval.X_test
    data_train.y_test = data_eval.y_test
    data_train.ids_test = data_eval.ids_test
    
    del(data_eval)
    
    ## preprocessors
    fpath = os.path.join(
        args['artifacts_fpath'],
        f"analysis_id={args['analysis_id']}",
        args['preprocessors_fpath'],
    )
    
    f = open(f"{fpath}/{args['train_group']}.pkl","rb")
    pipe = pickle.load(f)
    
    # preprocess datasets using trained preprocessors
    data_train.X_train = pipe.transform(data_train.X_train)
    data_train.X_val = pipe.transform(data_train.X_val)
    data_train.X_test = pipe.transform(data_train.X_test)
    
    # get torch loaders
    loaders = data_train.to_torch()
    
    del(data_train)
    
    return loaders
    
def get_model_fnames(args):
    """
    Extract a list of available model names
    """
    fpath = os.path.join(
        args['artifacts_fpath'],
        f"analysis_id={args['analysis_id']}",
        "models"   
    )
    fnames = os.listdir(fpath)

    model_fname = args['train_group']
    
    model_fnames = [
        x for x in fnames if 
        '_'.join(x.split('_')[1:-1]) == model_fname
    ]
    return model_fnames
    
def get_model(args, model_fname, loaders):
    """
    Get DNN with selected hyperparameteres & load weights based on args['i_iter']
    """ 
    fname = '_'.join([
        'nn',
        args['analysis_id'],
        args['train_group'],
        args['analysis_tag']
    ])
    
    fpath = "/hpf/projects/lsung/projects/public/mimic4ds_public/mimic4ds/Experiments/baseline/hyperparams/models"
    
    # load nn params
    f = open(f"{fpath}/{fname}/model_params.yml",'r')
    param = yaml.load(f,Loader=yaml.FullLoader)
    param['input_dim'] = next(iter(loaders['test']))['features'].shape[1]
    m = FixedWidthModel(**param)

    fpath = os.path.join(
        args['artifacts_fpath'],
        f"analysis_id={args['analysis_id']}",
        "models",
        model_fname
    )
    
    # Load nn weights
    m.load_weights(f"{fpath}/model_weights")
    
    return m


def evaluate(args, df):
    """bootstrap evaluate models to obtain mean & std across models
    at each iteration
    """
    df_results = pd.DataFrame()
    evaluator = StandardEvaluator()
    
    # bootstrap
    for i in range(args['n_boot']):
        i_df = df.groupby(["phase", "iter"]).sample(frac=1, replace=True)
        
        i_metrics = evaluator.evaluate(
            i_df,
            strata_vars = ['phase', 'iter']
        )
        
        i_mean = i_metrics.groupby(["phase","metric"])['performance'].agg(
            ['mean','std']
        ).reset_index()
        
        i_mean['boot_iter'] = i+1
        df_results = pd.concat((df_results, i_mean),axis=0)
    
    return df_results

def evaluate_ensemble(args,df):
    """bootstrap evaluate average model outputs 
    to obtain ensemble performance
    """
    df_results = pd.DataFrame()
    evaluator = StandardEvaluator()
    
    # bootstrap
    for i in range(args['n_boot']):
        i_df = df.groupby(["phase","iter"]).sample(frac=1,replace=True)
        i_df = i_df.groupby(["phase","row_id"]).mean().reset_index()
        
        i_metrics = evaluator.evaluate(
            i_df, 
            strata_vars = ['phase'],
        )
        
        i_metrics['boot_iter'] = i+1
        df_results = pd.concat((df_results, i_metrics),axis=0)
        
    return df_results
        
def evaluate_best(args,df):
    """bootstrap evaluate best model determined based on 
    validation performance
    """
    df_results = pd.DataFrame()
    evaluator = StandardEvaluator()
    
    # get best model based on validation bce loss
    performance = evaluator.evaluate(
        df.query("phase=='val'"), 
        strata_vars=['iter']
    )
    
    idx_best_model = performance.groupby(
        'metric'
    )['performance'].idxmin()['loss_bce']
    
    best_iter = performance.loc[idx_best_model,'iter']
    df = df.query("iter==@best_iter")
    
    # bootstrap
    for i in range(args['n_boot']):
        i_df = df.groupby('phase').sample(frac=1, replace=True)
        
        i_metrics = evaluator.evaluate(
            i_df,
            strata_vars=['phase']
        )
        
        i_metrics['boot_iter'] = i+1
        df_results = pd.concat((df_results, i_metrics), axis=0)
    
    return df_results


'''
Deprecated

def get_threshold(df,q):
    """
    Obtain risk threshold based on quantile in validation set
    """
    df_c = df.copy()
    df_c = df_c.query("phase=='val'")
    
    df_c['threshold'] = df_c.groupby('iter')[
        'pred_probs'
    ].transform(lambda x: x.quantile(1-q))
    
    df = pd.merge(
        df, 
        df_c.groupby('iter')['threshold'].mean().reset_index(), 
        how='left',
        left_on='iter',
        right_on='iter' 
    )
    
    return df
    

def get_optimal_threshold(df):
    """
    ***Deprecated***
    Obtain optimal threshold from validation set based on
    predictive values (PPV & NPV)
    """
    df_c = df.copy()
    evaluator = StandardEvaluator()
    
    df_c = df_c.query("phase=='val'")
    
    df_results = pd.DataFrame(dtype = float)
    
    # get threshold & threshold metrics based on top i% risk
    # scores in the validation set.
    for c,i in enumerate([0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]):
        
        df_c['threshold'] = df_c.groupby('iter')[
            'pred_probs'
        ].transform(lambda x: x.quantile(1-i))
        
        df_temp = evaluator.evaluate(
            df_c,strata_vars=['iter']
        ).query(
            "metric==['ppv','npv']"
        ).groupby('iter')['performance'].sum().reset_index()
        
        df_results= pd.concat((
            df_results, 
            pd.merge(
                df_temp,
                df_c.groupby('iter')['threshold'].mean().reset_index(),
                how='left',
                left_on='iter',
                right_on='iter',
            )
        )).reset_index(drop=True)
    
    # find threshold that maximized ppv + npv
    idx = df_results.groupby('iter')[
        'performance'
    ].idxmax().reset_index(drop=True)
    
    # add threshold to df
    df = pd.merge(
        df,
        df_results.loc[idx,:][['iter','threshold']],
        how = 'left',
        left_on = 'iter',
        right_on = 'iter',
    )
        
    return df
'''

# ---------------------------------------------
# run
# ---------------------------------------------
if __name__ == "__main__":
    
    args = vars(parser.parse_args())
    np.random.seed(args['seed'])
    
    ## print
    print('#####'*10)
    print(f"Evaluate nn trained on {args['train_group']}")
    print(f"Evaluating on {args['eval_group']} data")
    print(f"task:{args['analysis_id']}")
    print(f"n_boot: {args['n_boot']}")
    print('#####'*10)
    
    ##get pred_probs if files do not already exist
    fname = '_'.join(
        filter(None,[
            'nn',
            args['train_group'],
            args['eval_group']
        ])
    )
    fpath = os.path.join(
        args['artifacts_fpath'],
        f"analysis_id={args['analysis_id']}",
        'results',
        'pred_probs',
        f"{fname}.csv"
    )
    
    if os.path.exists(fpath):
        get_pred_probs = False
        outputs = pd.read_csv(fpath)
    else:
        get_pred_probs = True
        
    if get_pred_probs:
    
        # get data
        print('Loading Data...')
        loaders = get_data(args)

        # load all models to be evaluated
        model_fnames = get_model_fnames(args)
        print(f"Found {len(model_fnames)} models...")

        # loop through all models
        outputs = pd.DataFrame()
        c=0
        for model_fname in model_fnames:
            c+=1
            # get model
            print(f"getting outputs for {model_fname}")
            m = get_model(args, model_fname, loaders)
            # get model outputs
            i_outputs_val = m.predict(loaders, phases=['val'])['outputs']

            it = iter(loaders['val'])
            df = pd.DataFrame(dtype = float)
            for i in range(len(loaders['val'])):
                i_it = next(it)
                df = pd.concat((
                    df, pd.DataFrame({
                        key: val for key, val in i_it.items() if
                        key in ['row_id','ids']
                    })
                ))

            i_outputs_val = pd.merge(i_outputs_val, df, left_on = 'row_id', right_on = 'row_id')
            i_outputs_val['phase'] = 'val'
            i_outputs_val['iter'] = c

            # test
            i_outputs_test = m.predict(loaders, phases=['test'])['outputs']

            it = iter(loaders['test'])
            df = pd.DataFrame(dtype = float)
            for i in range(len(loaders['test'])):
                i_it = next(it)
                df = pd.concat((
                    df, pd.DataFrame({
                        key: val for key, val in i_it.items() if
                        key in ['row_id','ids']
                    })
                ))

            i_outputs_test = pd.merge(i_outputs_test, df, left_on = 'row_id', right_on = 'row_id')
            i_outputs_test['phase'] = 'test'
            i_outputs_test['iter'] = c

            outputs = pd.concat((outputs, i_outputs_val, i_outputs_test),axis=0).reset_index(drop=True)

        outputs['train_group'] = args['train_group']
        outputs['eval_group'] = args['eval_group']
        outputs['analysis_id'] = args['analysis_id']


        # save outputs
        print("saving outputs...")
        fpath = os.path.join(
            args['artifacts_fpath'],
            f"analysis_id={args['analysis_id']}",
            'results',
            'pred_probs'
        )
        os.makedirs(fpath, exist_ok = True)

        fname = '_'.join(
            filter(None,[
                'nn',
                args['train_group'],
                args['eval_group']
            ])
        )

        outputs.to_csv(f"{fpath}/{fname}.csv")
    
    if args['eval']:
        # bootstrap evaluate models
        print("evaluating outputs...")
        
        # run evaluation
        if args['evaluation_method'] == 'avg':
            df_results = evaluate(args, outputs)

        elif args['evaluation_method'] == 'ensemble':
            df_results = evaluate_ensemble(args, outputs)

        elif args['evaluation_method'] == 'best':
            df_results = evaluate_best(args, outputs)

        # add additional info to df
        df_results['train_group'] = args['train_group']
        df_results['eval_group'] = args['eval_group']
        df_results['analysis_id'] = args['analysis_id']

        # save evaluation
        print("saving evaluation...")
        fpath = os.path.join(
            args['artifacts_fpath'],
            f"analysis_id={args['analysis_id']}",
            'results',
            'evaluate_models'
        )
        os.makedirs(fpath, exist_ok = True)

        fname = '_'.join(
            filter(None, [
                'nn',
                args['train_group'],
                args['eval_group'],
                args['evaluation_method'],
            ])
        )

        df_results.to_csv(f"{fpath}/{fname}.csv")