"""
Collect results

1. bootstrap CI of model evaluations

2. bootstrap CI of model comparisons 
    - compare domain adapt models with 
      ERM models
"""

import numpy as np
import pandas as pd

import os
import argparse

from scipy import stats
from utils_prediction.utils import str2bool

# Init parser
parser = argparse.ArgumentParser(
    description = "Collect model evaluation and comparison results",
)

## args - with defaults
# evaluation related
parser.add_argument(
    "--analysis_ids", 
    nargs = "+",
    type = str,
    default = ['mortality','longlos','sepsis','invasivevent'],
    help = "Prediction tasks",
)

parser.add_argument(
    "--n_oods",
    nargs = "+",
    type = int,
    default = [100, 500, 1000, 1500],
    help = "Number of OOD samples used to train models"
)

parser.add_argument(
    "--evaluation_methods",
    nargs="+",
    type = str,
    default = ['avg','best','ensemble'],
    help = "how models were evaluated",
)

parser.add_argument(
    "--metrics",
    type = str,
    nargs="+",
    default = ['auc','auprc','ace_abs_logistic_log'],
    help = "which metrics to collect",
)

parser.add_argument(
    "--collect_model_evaluations",
    type = str2bool,
    default = 'true',
    help = "whether to collect model evaluation results",
)

parser.add_argument(
    "--collect_model_comparisons",
    type = str2bool,
    default = 'true',
    help = "whether to collect model comparisons results",
)

parser.add_argument(
    "--collect_model_evaluations_by_dem",
    type = str2bool,
    default = 'false',
    help = "whether to collect model evaluations by demographic results",
)

parser.add_argument(
    "--collect_model_comparisons_by_dem",
    type = str2bool,
    default = 'false',
    help = "whether to collect model evaluations by demographic results",
)

parser.add_argument(
    "--demographic_categories",
    nargs="+",
    default = ["gender"],
    help = "which demographic categories"
)

parser.add_argument(
    "--alpha",
    type = float,
    default = 0.01,
    help = "alpha threshold, default = 0.01"
)

# save related
parser.add_argument(
    "--save_path",
    type = str,
    default = "/hpf/projects/lsung/projects/public/mimic4ds_public/mimic4ds/Experiments/domain_adapt/results",
)

parser.add_argument(
    "--save_tag",
    type = str,
    default = "",
    help = "Tag added to the end of the filename",
)

# path related
parser.add_argument(
    "--artifacts_fpath",
    type = str,
    default = "/hpf/projects/lsung/projects/public/mimic4ds_public/mimic4ds/Experiments/domain_adapt/artifacts",
    help = "Where artifacts are stored"
)

#-------------------------------------------------------------------
# helper functions
#-------------------------------------------------------------------

def get_CI_evaluation(args,df,evaluation_method):
    
    p_lower = args['alpha']/2
    p_upper = 1 - (args['alpha']/2)
    
    # gather metadata
    phases = df['phase'].unique()
    train_method = df['train_method'].unique()[0]
    task = df['analysis_id'].unique()[0]
    lambd = df['lambda'].unique()[0]
    n_ood = df['n_ood'].unique()[0]
    
    # rename mean to performance if exists
    # mean = avg across model performance
    # performance = performance of best / ensemble model
    if 'mean' in df.columns:
        df.rename(
            columns={'mean':'performance'}, 
            inplace=True,
        )
    
    out = pd.DataFrame(columns = [
        'evaluation_method',
        'phase',
        'group',
        'n_ood',
        'metric',
        'ci_lower',
        'ci_med',
        'ci_upper',
        'train_method',
        'lambda',
        'analysis_id'
    ])
    
    c=0
    for phase in phases:
        groups = df.query("phase==@phase")['group'].unique()
        for group in groups:
            for metric in args['metrics']:
                c+=1

                i_df = df.query("phase==@phase and group==@group and metric==@metric")

                ci_lower = i_df['performance'].quantile(q=p_lower)
                ci_upper = i_df['performance'].quantile(q=p_upper)
                ci_med = i_df['performance'].quantile(q=0.5)

                out.loc[c,:]=[
                    evaluation_method,
                    phase,
                    group,
                    n_ood,
                    metric,
                    ci_lower,
                    ci_med,
                    ci_upper,
                    train_method,
                    lambd,
                    task,
                ]
    
    return out

def get_CI_evaluation_by_dem(args,df,demographic_category):
    
    p_lower = args['alpha']/2
    p_upper = 1 - (args['alpha']/2)
    
    # gather metadata
    phases = df['phase'].unique()
    train_method = df['train_method'].unique()[0]
    task = df['analysis_id'].unique()[0]
    n_ood = df['n_ood'].unique()[0]
    dem_cats = df[demographic_category].unique()
    
    # rename mean to performance if exists
    # mean = avg across model performance
    # performance = performance of best / ensemble model
    if 'mean' in df.columns:
        df.rename(
            columns={'mean':'performance'}, 
            inplace=True,
        )
    
    out = pd.DataFrame(columns = [
        'phase',
        'group',
        'n_ood',
        'metric',
        'ci_lower',
        'ci_med',
        'ci_upper',
        'train_method',
        'analysis_id',
        f"{demographic_category}"
    ])
    
    c=0
    for phase in phases:
        groups = df.query("phase==@phase")['group'].unique()
        for group in groups:
            for dem_cat in dem_cats:
                for metric in args['metrics']:
                    c+=1

                    i_df = df.query(
                        f"phase==@phase and group==@group and {demographic_category}==@dem_cat and metric==@metric"
                    )

                    ci_lower = i_df['performance'].quantile(q=p_lower)
                    ci_upper = i_df['performance'].quantile(q=p_upper)
                    ci_med = i_df['performance'].quantile(q=0.5)

                    out.loc[c,:]=[
                        phase,
                        group,
                        n_ood,
                        metric,
                        ci_lower,
                        ci_med,
                        ci_upper,
                        train_method,
                        task,
                        dem_cat
                    ]

    return out

def get_CI_comparison(args,df,evaluation_method,fname):
    
    # get lambda hyperparamter from fname
    if 'lambda' in fname:
        fname_split = fname.split('_')
        
        lambd = float(
            fname_split[fname_split.index('lambda')+1]
        )
        
    else:
        lambd = -1
    
    p_lower = args['alpha']/2
    p_upper = 1 - (args['alpha']/2)
    
    # gather metadata
    phases = df['phase'].unique()
    train_method = df['train_method'].unique()[0]
    task = df['analysis_id'].unique()[0]
    n_ood = df['n_ood'].unique()[0]
    
    out = pd.DataFrame(columns = [
        'evaluation_method',
        'phase',
        'group',
        'n_ood',
        'metric',
        'ci_lower',
        'ci_med',
        'ci_upper',
        'train_method',
        'lambda',
        'analysis_id',
    ])
    
    c=0
    for phase in phases:
        groups = df.query("phase==@phase")['group'].unique()
        for group in groups:
            for metric in args['metrics']:
                c+=1

                i_df = df.query("phase==@phase and group==@group and metric==@metric")

                ci_lower = i_df['performance_diff'].quantile(q=p_lower)
                ci_upper = i_df['performance_diff'].quantile(q=p_upper)
                ci_med = i_df['performance_diff'].quantile(q=0.5)

                out.loc[c,:]=[
                    evaluation_method,
                    phase,
                    group,
                    n_ood,
                    metric,
                    ci_lower,
                    ci_med,
                    ci_upper,
                    train_method,
                    lambd,
                    task,
                ]

        return out

def get_CI_comparison_by_dem(args,df,demographic_category):
    
    p_lower = args['alpha']/2
    p_upper = 1 - (args['alpha']/2)
    
    # gather metadata
    phases = df['phase'].unique()
    train_method = df['train_method'].unique()[0]
    task = df['analysis_id'].unique()[0]
    n_ood = df['n_ood'].unique()[0]
    dem_cats = df[demographic_category].unique()
    
    out = pd.DataFrame(columns = [
        'phase',
        'group',
        'n_ood',
        'metric',
        'ci_lower',
        'ci_med',
        'ci_upper',
        'train_method',
        'analysis_id',
        f"{demographic_category}"
    ])
    
    c=0
    for phase in phases:
        groups = df.query("phase==@phase")['group'].unique()
        for group in groups:
            for dem_cat in dem_cats:
                for metric in args['metrics']:
                    c+=1

                    i_df = df.query(
                        f"phase==@phase and group==@group and {demographic_category}==@dem_cat and metric==@metric"
                    )

                    ci_lower = i_df['performance_diff'].quantile(q=p_lower)
                    ci_upper = i_df['performance_diff'].quantile(q=p_upper)
                    ci_med = i_df['performance_diff'].quantile(q=0.5)

                    out.loc[c,:]=[
                        phase,
                        group,
                        n_ood,
                        metric,
                        ci_lower,
                        ci_med,
                        ci_upper,
                        train_method,
                        task,
                        dem_cat,
                    ]

        return out



#-------------------------------------------------------------------
# run 
#-------------------------------------------------------------------
if __name__ == "__main__":
    
    args = vars(parser.parse_args())
    
    # collect model evaluation results
    if args['collect_model_evaluations']:
        
        print("Collecting model evaluation results...")
        
        df_results = pd.DataFrame()
        
        for analysis_id in args['analysis_ids']:
            for method in args['evaluation_methods']:
                
                fpath = os.path.join(
                    args['artifacts_fpath'],
                    f"analysis_id={analysis_id}",
                    "results/evaluate_models"
                )
                
                fnames = os.listdir(fpath)
                
                df_results = pd.concat((
                    df_results,
                    pd.concat((
                        get_CI_evaluation(
                            args,
                            pd.read_csv(f"{fpath}/{x}"),
                            method
                        )
                        for x in fnames
                        if method in x and
                        'group_id' in x
                    ))
                ))
                
        os.makedirs(args['save_path'], exist_ok = True)
        
        fname = '_'.join(
            filter(None, [
                "model_evaluation",
                args['save_tag']
            ])
        )
        
        save_path = os.path.join(
            args['save_path'],
            f"{fname}.csv"
        )
        
        df_results.to_csv(save_path)
    
    # collect model comparison results
    if args['collect_model_comparisons']:
        
        print("Collecting model comparison results...")
        
        df_results = pd.DataFrame()
        
        for analysis_id in args['analysis_ids']:
            for method in args['evaluation_methods']:
                
                fpath = os.path.join(
                    args['artifacts_fpath'],
                    f"analysis_id={analysis_id}",
                    "results/compare_models"
                )
                
                fnames = os.listdir(fpath)
                
                df_results = pd.concat((
                    df_results,
                    pd.concat((
                        get_CI_comparison(
                            args,
                            pd.read_csv(f"{fpath}/{x}"),
                            method,
                            x
                        )
                        for x in fnames
                        if method in x and 
                        'group_id' in x
                    ))
                ))
                
        os.makedirs(args['save_path'], exist_ok = True)
        
        fname = '_'.join(
            filter(None, [
                "model_comparison",
                args['save_tag']
            ])
        )
        
        save_path = os.path.join(
            args['save_path'],
            f"{fname}.csv"
        )
        
        df_results.to_csv(save_path)
    
    # Collect model evaluation results by demographic
    if args['collect_model_evaluations_by_dem']:
        
        print("Collecting model evaluation results by demographic...")
        
        df_results = pd.DataFrame()
        
        for demographic_category in args['demographic_categories']:
            for analysis_id in args['analysis_ids']:
        
                fpath = os.path.join(
                    args['artifacts_fpath'],
                    f"analysis_id={analysis_id}",
                    "results",
                    "by_demographic",
                    "evaluate_models"
                )

                fnames = os.listdir(fpath)

                df_results = pd.concat((
                    df_results,
                    pd.concat((
                        get_CI_evaluation_by_dem(
                            args,
                            pd.read_csv(f"{fpath}/{x}"),
                            demographic_category
                        )
                        for x in fnames
                        if demographic_category in x
                    ))
                ))
                
            os.makedirs(args['save_path'], exist_ok = True)

            fname = '_'.join(
                filter(None, [
                    "model_evaluation",
                    "by",
                    demographic_category,
                    args['save_tag']
                ])
            )
        
            save_path = os.path.join(
                args['save_path'],
                f"{fname}.csv"
            )

            df_results.to_csv(save_path)
            
            
    # Collect model comparison results by demographic
    if args['collect_model_comparisons_by_dem']:
        
        print("Collecting model comparison results by demographic...")
        
        df_results = pd.DataFrame()
        
        for demographic_category in args['demographic_categories']:
            for analysis_id in args['analysis_ids']:
        
                fpath = os.path.join(
                    args['artifacts_fpath'],
                    f"analysis_id={analysis_id}",
                    "results",
                    "by_demographic",
                    "compare_models"
                )

                fnames = os.listdir(fpath)

                df_results = pd.concat((
                    df_results,
                    pd.concat((
                        get_CI_comparison_by_dem(
                            args,
                            pd.read_csv(f"{fpath}/{x}"),
                            demographic_category
                        )
                        for x in fnames
                        if demographic_category in x
                    ))
                ))
                
            os.makedirs(args['save_path'], exist_ok = True)

            fname = '_'.join(
                filter(None, [
                    "model_comparison",
                    "by",
                    demographic_category,
                    args['save_tag']
                ])
            )
        
            save_path = os.path.join(
                args['save_path'],
                f"{fname}.csv"
            )

            df_results.to_csv(save_path)