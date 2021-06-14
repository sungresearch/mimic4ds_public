import argparse
import yaml
import pickle

from utils_prediction.model_evaluation import StandardEvaluator
from utils_prediction.dataloader.mimic4 import *
from utils_prediction.utils import str2bool

parser = argparse.ArgumentParser(description = "Compare models trained \
using ERM under domain generalization framework with models in the baseline \
Experiment")

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
    help="method used to train model ['erm','irm','dro','coral','al_layer']"
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

parser.add_argument(
    "--baseline_artifacts_fpath",
    type = str,
    default = "/hpf/projects/lsung/projects/public/mimic4ds_public/mimic4ds/Experiments/baseline/artifacts",
    help = "general path where datasets, preprocessors, \
    models, and results of oracle models are stored"
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
    "--comparison_model",
    type = str,
    default = 'base',
    help = "Whether to compare ERM results with 'base' \
    [models trained on 08 - 10] results or 'oracle' [models \
    trained on 17 - 19] results"
)

parser.add_argument(
    "--n_ood", 
    type = int, 
    required = True, 
    help="number of OOD samples used to train the model \
    [100, 500, 1000, 1500]"
)

# ---------------------------------------------
# helper funcs
# ---------------------------------------------
def compare(args, df_base, df_test):
    """bootstrap compare models to obtain difference between means 
    at each iteration
    
    df_base: results of models trained using erm
    df_test: results of models from baseline experiment (08 - 10 or oracle)
    """
    
    # select just the test phase [val phases are different b/w
    # domain generalization and baseline experiments]
    df_base = df_base.query("phase=='test'").sort_values(by=['iter','ids'],ignore_index=True)
    df_test = df_test.query("phase=='test'").sort_values(by=['iter','ids'],ignore_index=True)
    
    df_results = pd.DataFrame()
    evaluator = StandardEvaluator()
    
    # bootstrap
    for i in range(args['n_boot']):
        
        i_df_base = df_base.groupby(["iter"]).sample(frac=1, replace=True)
        i_df_test = df_test.loc[i_df_base.index,:]
        
        i_metrics_base = evaluator.evaluate(
            i_df_base,
            strata_vars = ['iter']
        )
        
        i_metrics_test = evaluator.evaluate(
            i_df_test,
            strata_vars = ['iter']
        )
        
        i_mean_base = i_metrics_base.groupby(
            ["metric"]
        )['performance'].mean().reset_index()
        
        i_mean_test = i_metrics_test.groupby(
            ["metric"]
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

# ---------------------------------------------
# run
# ---------------------------------------------
if __name__ == "__main__":
    
    args = vars(parser.parse_args())
    np.random.seed(args['seed'])
    
    ## print
    print('#####'*10)
    print(f"Compare domain adaptation models (base) w/ baseline models (test)")
    print(f"task:{args['analysis_id']}")
    print(f"n_boot: {args['n_boot']}")
    print('#####'*10)
    
    # get model outputs 
    fpath_base = os.path.join(
        args['artifacts_fpath'],
        f"analysis_id={args['analysis_id']}",
        "results",
        "pred_probs",
    )
    
    fpath_test = os.path.join(
        args['baseline_artifacts_fpath'],
        f"analysis_id={args['analysis_id']}",
        "results",
        "pred_probs",
    )
    
    fname_base = '_'.join(
        filter(None,[
            "nn",
            args['train_method'],
            str(args['n_ood']),
        ])
    )
    
    fname_test = '_'.join(
        filter(None,[
            "nn",
            '2008 - 2010' if args['comparison_model']=='base' else
            '2017 - 2019',
            '2017 - 2019'
        ])
    )
    
    print("Loading model outputs...")

    df_base = pd.read_csv(f"{fpath_base}/{fname_base}.csv").query("\
    group==3\
    ")[
        ['phase','outputs','pred_probs','labels','row_id','ids','iter']
    ]
    
    df_test = pd.read_csv(f"{fpath_test}/{fname_test}.csv")[
        ['phase','outputs','pred_probs','labels','row_id','ids','iter']
    ]
    
    # Check that ids are equal across dfs to be compared
    row_id_check = np.intersect1d(
        df_base.query("phase=='test'").reset_index(drop=True)[
            'ids'
        ].unique(), 
        df_test.query("phase=='test'").reset_index(drop=True)[
            'ids'
        ].unique()
    )
    
    assert(len(row_id_check)==len(df_base.query("phase=='test'")['ids'].unique())), (
        'IDs between results need to be equal in order to compare')
    
    # bootstrap evaluate models
    print("Comparing model performance...")

    df_results = compare(args, df_base, df_test)
    
    # add additional info to df
    df_results['train_method'] = args['train_method']
    df_results['n_ood'] = args['n_ood']
    df_results['comparison_model'] = args['comparison_model']
    df_results['analysis_id'] = args['analysis_id']
    
    # save evaluation
    print("saving evaluation...")
    fpath = os.path.join(
        args['artifacts_fpath'],
        f"analysis_id={args['analysis_id']}",
        'results',
        'compare_with_baseline_exp',
    )
    os.makedirs(fpath, exist_ok = True)
    
    fname = '_'.join(
        filter(None, [
            'nn',
            args['train_method'],
            str(args['n_ood']),
            args['comparison_model']
        ])
    )
    
    df_results.to_csv(f"{fpath}/{fname}.csv")