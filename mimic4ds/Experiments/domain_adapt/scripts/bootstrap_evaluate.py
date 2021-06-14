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
    ['al_layer','coral']"
)

## args - with defaults
# general
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
    "--evaluation_method",
    type = str,
    default = "avg",
    help = "Whether to bootstrap evaluate avg, \
    ensemble, or best model performance"
)

# additional 
parser.add_argument(
    "--lambd",
    type = float,
    default = -1,
    help = "if > 0, look for lambda sweep results"
)

parser.add_argument(
    "--group_train_domains",
    type = str2bool,
    default = 'true',
    help = 'whether to group all training domains as the same.'
)

parser.add_argument(
    "--eval",
    type = str2bool,
    default = 'true',
    help = "whether to evaluate model outputs"
)

# ---------------------------------------------
# helper funcs
# ---------------------------------------------
def get_data(args):
    """
    Load data using args['eval_group']
    Preprocess data using preprocessors trained on args['train_group']
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
    ).load_datasets(fname=f"preprocessed_dense_{args['n_ood']}")
    
    # get torch loaders
    loaders = data.to_torch(group_var_name='group_var')
    
    del(data)
    
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
    
    if args['lambd']>0:
        model_fname = f"{args['train_method']}_{args['n_ood']}_lambda_{args['lambd']}"
    else:
        model_fname = f"{args['train_method']}_{args['n_ood']}"
    
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
        args['analysis_id']
    ])
    
    fpath = "/hpf/projects/lsung/projects/public/mimic4ds_public/mimic4ds/Experiments/domain_gen/hyperparams/models"
    
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
        i_df = df.groupby(["phase", "group", "iter"]).sample(frac=1, replace=True)
        
        i_metrics = evaluator.evaluate(
            i_df,
            strata_vars = ['phase', 'group', 'iter']
        )
        
        i_mean = i_metrics.groupby(["phase", "group", "metric"])['performance'].agg(
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
        i_df = df.groupby(["phase","group","iter"]).sample(frac=1,replace=True)
        i_df = i_df.groupby(["phase","group","row_id"]).mean().reset_index()
        
        i_metrics = evaluator.evaluate(
            i_df, 
            strata_vars = ['phase','group'],
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
        i_df = df.groupby(['phase','group']).sample(frac=1, replace=True)
        
        i_metrics = evaluator.evaluate(
            i_df,
            strata_vars=['phase','group']
        )
        
        i_metrics['boot_iter'] = i+1
        df_results = pd.concat((df_results, i_metrics), axis=0)
    
    return df_results

# ---------------------------------------------
# run
# ---------------------------------------------
if __name__ == "__main__":
    
    args = vars(parser.parse_args())
    np.random.seed(args['seed'])
    
    ## print
    print('#####'*10)
    print(f"Evaluate nn trained under {args['train_method']}")
    print(f"task:{args['analysis_id']}")
    print(f"n_boot: {args['n_boot']}")
    print('#####'*10)
    
    # check if pred_prob file exists:
    fname = '_'.join(
        filter(None,[
            'nn',
            args['train_method'],
            str(args['n_ood']),
            f"lambda_{args['lambd']}" if args['lambd']>0 else "",
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

            ## get model outputs
            # val
            i_outputs_val = m.predict(loaders, phases=['val'])['outputs']

            it = iter(loaders['val'])
            df = pd.DataFrame(dtype = float)
            for i in range(len(loaders['val'])):
                i_it = next(it)
                df = pd.concat((
                    df, pd.DataFrame({
                        key: val for key, val in i_it.items() if
                        key in ['row_id','group','ids']
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
                        key in ['row_id','group','ids']
                    })
                ))

            i_outputs_test = pd.merge(i_outputs_test, df, left_on = 'row_id', right_on = 'row_id')
            i_outputs_test['phase'] = 'test'
            i_outputs_test['iter'] = c

            outputs = pd.concat((outputs, i_outputs_val, i_outputs_test),axis=0)

        outputs['train_method'] = args['train_method']
        outputs['analysis_id'] = args['analysis_id']
        outputs['n_ood'] = args['n_ood']
        if args['lambd']>0:
            outputs['lambda_sweep'] = 1
            outputs['lambda'] = args['lambd']
        else:
            outputs['lambda_sweep'] = 0
            outputs['lambda'] = -1

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
                args['train_method'],
                str(args['n_ood']),
                f"lambda_{args['lambd']}" if args['lambd']>0 else "",
            ])
        )

        outputs.to_csv(f"{fpath}/{fname}.csv")
    
    if args['eval']:
        ## bootstrap evaluate models
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
        if args['evaluation_method'] == 'avg':
            df_results = evaluate(args, outputs)

        elif args['evaluation_method'] == 'ensemble':
            df_results = evaluate_ensemble(args, outputs)

        elif args['evaluation_method'] == 'best':
            df_results = evaluate_best(args, outputs)

        # add additional info to df
        df_results['train_method'] = args['train_method']
        df_results['analysis_id'] = args['analysis_id']
        df_results['n_ood'] = args['n_ood']
        if args['lambd']>0:
            df_results['lambda_sweep'] = 1
            df_results['lambda'] = args['lambd']
        else:
            df_results['lambda_sweep'] = 0
            df_results['lambda'] = -1
        df_results['task'] = args['analysis_id']

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
                args['train_method'],
                str(args['n_ood']),
                f"lambda_{args['lambd']}" if args['lambd']>0 else "",
                "group_id" if args['group_train_domains'] else "",
                args['evaluation_method'],
            ])
        )

        df_results.to_csv(f"{fpath}/{fname}.csv")