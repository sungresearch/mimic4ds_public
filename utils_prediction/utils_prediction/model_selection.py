"""
Python classes that help with model selection 
"""


import os
import yaml
import pickle
import torch

import numpy as np

from joblib import Parallel, delayed
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import (
    roc_auc_score, 
    precision_recall_curve, 
    accuracy_score, 
    f1_score, 
    precision_score, 
    recall_score, log_loss
)

class gridsearch():
    """
    conducts grid search for classification using training and validation sets
    
    -----------
    Parameters
    -----------
    model: sklearn model with fit_transform
    
    param_grid: dictionary with keys being parameter type and values being parameters
        e.g., {'n_estimators':[10,50]}
    
    additional arguments:
        save_fpath: path to save model & hyperparameters
        save_model: boolean (default = False)
        save_params: boolean (default = False)
        n_jobs: number of parallel search (for sklearn models / cpu torch model)
        random_search: conduct random grid search (default: True)
        n_searches: number of searches for random_search (default: 50)
        fit_params: a dictionary with additional parameters to be passed onto model.fit()
        
    available metrics:
        auroc, auprc, accuracy, f1_score, precision, recall, log_loss [default: log_loss]
        
    """
    
    def __init__(self, model, param_grid, *args, **kwargs):
        self.config_dict = self.__get_default_config()
        self.config_dict = self.__override_config(**kwargs)
        self.model = model
        self.param_grid = self.__get_param_grid(param_grid)
        
        self.metrics = {
            'auroc':roc_auc_score,
            'auprc':precision_recall_curve,
            'accuracy':accuracy_score,
            'f1':f1_score,
            'precision':precision_score,
            'recall':recall_score,
            'log_loss':log_loss
       }
        
        self.best_param = None
        self.best_score = 0.
    
    def __get_default_config(self):
        """
        Defines default gridsearch settings
        """
        return {
            "save_model":False,
            "save_weights":False,
            "random_search":True,
            "n_searches":50,
            "n_jobs":1,
            "save_fpath":'/hpf/projects/lsung/projects/mimic4ds/artifacts/',
            "fit_params":{}
        }
    
    def __override_config(self,**override_dict):
        """
        Updates the config_dict with elements of override_dict
        """
        return {**self.config_dict, **override_dict}
    
    def __get_param_grid(self,param_grid):
        """
        Generates param_grid based on param_grid and config
        """
        param_grid = list(ParameterGrid(param_grid))
        if self.config_dict['random_search']:
            np.random.shuffle(param_grid)
            if self.config_dict['n_searches']<len(param_grid):
                param_grid = param_grid[:self.config_dict['n_searches']]
        return param_grid
    
    def fit_helper(self, param, X_train, Y_train, X_validation, Y_validation, metric):
        """
        helper function to fit a model using user-defined param & fit_params
        """
        classifier = self.model(**param).fit(X_train, Y_train, **self.config_dict['fit_params'])
        m = self.metrics[metric]
        if (metric == 'auroc') | (metric == 'auprc') | (metric == 'log_loss'):
            Y_predict = classifier.predict_proba(X_validation)
            score = m(Y_validation, Y_predict[:,1])
        else:
            Y_predict = classifier.predict(X_validation)
            score = m(Y_validation, Y_predict)
        print('<{}> with param {} has {} of {}.'.format(classifier, param, metric, score))
        return (param, score)
    
    def fit(self, X_tr, Y_tr, X_val, Y_val, metric = 'log_loss'):
        """
        conduct parralel grid search, save and return the model with best hyperparameters
        """
        print(f"Performing {len(self.param_grid)} hyperparam searches in parallel using {self.config_dict['n_jobs']} jobs.")

        param_scores = Parallel(n_jobs=self.config_dict['n_jobs'])(
            delayed(self.fit_helper)
            (param, X_tr, Y_tr, X_val, Y_val,metric) 
            for param in self.param_grid
        )
        
        if metric != 'log_loss':
            self.best_param, self.best_score = max(param_scores, key=lambda x: x[1])
        elif metric == 'log_loss':
            self.best_param, self.best_score = min(param_scores, key=lambda x: x[1])
        
        print(f"Best scoring param is {self.best_param} with metric {metric}:{self.best_score}.")
        
        # fit model with best params
        self.model = self.model(**self.best_param).fit(X_tr, Y_tr)
        
        ## Save model
        if self.config_dict['save_model'] or self.config_dict['save_params']:
            self.save()
        
        return self.model
    
    def save(self):
        # Create path if does not exist
        dir_path = self.config_dict['save_fpath']
        
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        
        if self.config_dict['save_model']:
            fpath = dir_path+'/model.pkl'
            print('saving model to',fpath)
            f = open(fpath, 'wb')
            pickle.dump(self.model,f)
        if self.config_dict['save_params']:
            f = open(dir_path+'/model_params.yml','w')
            yaml.dump(self.best_param,f)
    

class gridsearch_nn(gridsearch):
    """
    Gridsearch built for torch models implemented in the nn module that inherits gridsearch class
    
    -----------
    Parameters
    -----------
    model: torch model from utils_prediction.nn.models
    
    param_grid: dictionary with keys being parameter type and values being parameters
    
    additional arguments:
        save_fpath: path to save model
        save_fname: filename to be saved to path
        save: boolean (default = True)
        n_jobs: number of parallel search
        random_search: conduct random grid search (default: True)
        n_searches: number of searches for random_search (default: 50)
        fit_params: a dictionary with additional fit parameters to be passed onto model.fit()
        
        metric is defined at the model / grid level
        default = loss, unless otherwise specified using "selection_metric" in the hyperparameter grid
    """
    
    def fit_helper(self, param, loaders, phases):
        """
        helper function to fit a model using user-defined param & fit_param
        """
        input_dim = next(iter(loaders['train']))['features'].shape[1]
        param['input_dim'] = input_dim
        
        m = self.model(**param)
        self.metric = m.config_dict['selection_metric']
        
        _ = m.train(loaders,phases=phases)
        
        df = m.predict(loaders,phases=['val'])['performance']
        score = df.loc[df['metric']==self.metric]['performance'].values[0]
        
        print('<{}> with param {} has {} of {}.'.format(m, param, self.metric, score))
        return (param, score)
    
    def fit(self, loaders, phases = ['train','val']):
        """
        conduct parralel grid search, save and return the model with best hyperparameters
        """
        print(f"Performing {len(self.param_grid)} hyperparam searches in parallel using {self.config_dict['n_jobs']} jobs.")

        # if GPU available - use it; else use multi-core CPU
        if torch.cuda.is_available():
            n_jobs = 1
        else:
            n_jobs = self.config_dict['n_jobs']
        
        param_scores = Parallel(n_jobs=n_jobs)(
            delayed(self.fit_helper)
            (param, loaders, phases) 
            for param in self.param_grid
        )
        
        if self.metric == 'loss' or self.metric == 'brier' or self.metric=='supervised':
            self.best_param, self.best_score = min(param_scores, key=lambda x: x[1])
        else:
            self.best_param, self.best_score = max(param_scores, key=lambda x: x[1])
        print(f"Best scoring param is {self.best_param} with metric {self.metric}:{self.best_score}.")
        
        # fit model with best params
        self.model = self.model(**self.best_param)
        _ = self.model.train(loaders, phases = phases)
        
        
        # Save model
        if self.config_dict['save_model'] or self.config_dict['save_params']:
            self.save()
        
        # return model
        return self.model
            
    def save(self):
        """
        Saves model weights and config (yaml) into a folder under artifacts/models/save_fname
        """
        # Create path if does not exist
        fpath = self.config_dict['save_fpath']

        if not os.path.exists(fpath):
            os.makedirs(fpath)
        
        if self.config_dict['save_model']:
            print('saving model weights to',fpath)
            self.model.save_weights(fpath+'/model_weights')
            
        if self.config_dict['save_params']:
            print('saving selected model hyperparameters to',fpath)
            f = open(fpath+'/model_params.yml',"w")
            yaml.dump(self.best_param,f)