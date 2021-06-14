"""
Feature extraction from GBQ based on configurations in feature_configs
"""
import yaml
import argparse

from utils_prediction.database import *
from utils_prediction.featurizer.mimic4_gbq import *

## Init parser
parser = argparse.ArgumentParser(
    description = "Extract features based on configurations featurizer_configs"
)

## args
parser.add_argument(
    "--save_path",
    type = str,
    required = True,
    help="Where to save extracted features to"
)

parser.add_argument(
    "--config_path", 
    type = str, 
    required = True, 
    help="path to yaml config file"
)


### run
if __name__=="__main__":
    
    args = vars(parser.parse_args())
    
    # load config
    f = open(f"{args['config_path']}","r")
    config = yaml.load(f, Loader = yaml.FullLoader)
    config['save_fpath'] = args['save_path']
    
    # run featurizer
    featurizer(config).featurize_and_save()
    
    print('done')
