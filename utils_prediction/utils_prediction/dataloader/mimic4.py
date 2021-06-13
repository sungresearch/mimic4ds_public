import os
import pickle

import pandas as pd
import numpy as np 
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.feather as feather

from scipy.sparse import csr_matrix, vstack
from sklearn.model_selection import StratifiedShuffleSplit
from utils_prediction.nn.datasets import ArrayLoaderGenerator


class dataloader():
    """
    dataloader class for extracted mimic4 features 
    
    split functions:
    -------------------
        split() - stratified split
        split_by_ids() - supports custom splitting via id file (CSV)
    
    kwargs:
    -------------------
        analysis_id: string [e.g., 'mortality']
        features_fpath: string [e.g., path to extracted MIMIC4 features]
        features_ftype: string [default: 'parquet']
        datasets_fpath: string [e.g., path to split datasets]
        datasets_ftype: string [default: 'parquet']
        label_col: string [use 'label']
        id_col: string [use 'subject_id']
        verbose: bool [default: False]
        group:  extracts a particular group from group_var. 
                "group_var" must exist as a column in feature 
                dataframe
          
    """
    
    def __init__(self, *args, **kwargs):
        
        self.config = self.__build_default_config()
        self.config = self.__update_config(**kwargs)
        
        if 'id_col' not in self.config: 
            raise UserWarning('Please specify ID column')
            
        if 'label_col' not in self.config: 
            raise UserWarning('Please specify label column')
    
    def load_features(self):
        """
        Loads extracted features.
        Requires:
            analysis_id
            features_path
            features_ftype
            label_col
            id_col
        
        Optional:
            group
        """
        # Load features
        if self.config['features_ftype'] == 'parquet':
            self.features = pq.read_table(
                os.path.join(
                    self.config['features_fpath'],
                    f"analysis_id={self.config['analysis_id']}",
                    "features.parquet"
                )
            ).to_pandas()
            
        elif self.config['features_ftype'] == 'feather':
            self.features = feather.read_feather(
                os.path.join(
                    self.config['features_fpath'],
                    f"analysis_id={self.config['analysis_id']}",
                    "features.feather"
                )
            )
        
        # If group is specified, extract only group rows
        if (
            'group' in self.config and 
            'group_var' in self.features.columns
        ): 
            
            self.features = self.features.query(
                "group_var == @self.config['group']"
            ).reset_index(drop=True)
                
        if self.config['verbose']: 
            print('\n',
                  len(self.features.columns),
                  'columns\n',
                  len(self.features),
                  'rows\n',
                  round(
                      np.sum(self.features.memory_usage())/1000000,
                      2
                  ),
                  'MBs in memory\n',
                 )
        
        # rename age to age_measurement for downstream processing 
        # of measurement data
        if 'age' in self.features.columns:
            self.features.rename(
                columns={
                    'age':'age_measurement'
                }, 
                inplace = True,
            )
        
        return self
    
    def split(
        self, 
        p_splits=[0.7,0.15,0.15], 
        seed = 44, 
        remove_original = True, 
        split_type = 'stratified', 
        retain_group_var = False
    ):
        """
        conduct stratified splitting
        """
        if split_type == 'stratified':
            
            return self.__split_stratified(
                p_splits, 
                seed, 
                remove_original, 
                retain_group_var
            )
        
        else:
            print('Not implemented, use split_type = stratified')
    
    def split_by_ids(
        self,
        ids_path = None,
        df_ids = None,
        id_col = 'subject_id',
        fold_id_col = 'fold_id',
        remove_original = True, 
        retain_group_var = False
    ):
        """
        split data according to ids in csv
        
        csv file must contain the following:
            columns:
                id_col,
                fold_id_col,
                
            fold_ids:
                train,
                val,
                test,
        """
        if df_ids is None:
            df_ids = pd.read_csv(ids_path)
        
        # check 1: train, val, and test folds exist
        if np.sum([
            x in df_ids['fold_id'].unique() 
            for x in ['train','val','test']
        ])!=3:
            raise UserWarning(f'need train,val,test in {fold_id_col}')
        
        # check 2: features have been loaded
        if hasattr(self, "features") == False:
            raise Exception("load_features() first")
        
        # check 3: unique ids across folds
        if not any([
            np.intersect1d(
                df_ids.query("fold_id=='train'")['subject_id'].values,
                df_ids.query("fold_id=='val'")['subject_id'].values,
            ).shape[0]==0,
            np.intersect1d(
                df_ids.query("fold_id=='train'")['subject_id'].values,
                df_ids.query("fold_id=='test'")['subject_id'].values,
            ).shape[0]==0,
            np.intersect1d(
                df_ids.query("fold_id=='test'")['subject_id'].values,
                df_ids.query("fold_id=='val'")['subject_id'].values,
            ).shape[0]==0,
        ]):
            raise Exception(f"Overlapping {id_col} across {fold_id_col}")
        
        # grab ids
        ids = {
            k:df_ids.query(f"{fold_id_col}==@k")[id_col].tolist()
            for k in ['train','val','test']
        }
        
        # assign datasets
        self.X_train = self.features.query(
            f"{self.config['id_col']}==@ids['train']"
        ).reset_index(
            drop=True
        )
        
        self.y_train = self.X_train.pop(self.config['label_col']).values
        self.ids_train = self.X_train.pop(self.config['id_col']).values
        
        self.X_val = self.features.query(
            f"{self.config['id_col']}==@ids['val']"
        ).reset_index(
            drop=True
        )
        
        self.y_val = self.X_val.pop(self.config['label_col']).values
        self.ids_val = self.X_val.pop(self.config['id_col']).values
        
        self.X_test = self.features.query(
            f"{self.config['id_col']}==@ids['test']"
        ).reset_index(
            drop=True
        )
        
        self.y_test = self.X_test.pop(self.config['label_col']).values
        self.ids_test = self.X_test.pop(self.config['id_col']).values
        
        # remove group_var if not retain
        if not retain_group_var:
                
            self.X_train = self.X_train.drop(
                columns=['group_var']
            )

            self.X_val = self.X_val.drop(
                columns=['group_var']
            )

            self.X_test = self.X_test.drop(
                columns=['group_var']
            )
        
        return self
        
    
    def sample(
        self, 
        n_samples = 2000, 
        n_features = 500
    ):
        """
        randomly select n_samples and n_features from df
        n_features will always include the first 5 features 
        + n_features - 5 randomly selected ones
        """
        cols = list(self.features.columns[:5])
        a = list(self.features.columns[6:])
        np.random.shuffle(a)
        
        return self.features[
            cols+a[:n_features-5]
        ].sample(
            n=n_samples
        ).reset_index(
            drop=True
        )
        
    def to_sparse(
        self, 
        sets = ['X_train','X_val','X_test'], 
        group_var_name = None
    ):
        """
        This will turn dataframes (specified in sets) in the dataloader object
        into dictionaries with the following keys:
        data.train: {
        'X': sparse matrix of features (previuosly X_train)
        'y': label array (y_train)
        'ids': subject_id array (previously ids_train)
        'cols': column names (previously X_train.columns)
        }
        
        This requires that the data are already preprocessed and contains no missing values.
        """
        if (hasattr(self, 'X_train')) & ('X_train' in sets):
            if self.__has_nas(self.X_train)==False:
                
                if group_var_name is not None:
                    g_tmp = self.X_train[group_var_name].values
                    self.X_train = self.X_train.drop(columns=group_var_name)
                    self.train = {
                        'X': csr_matrix(self.X_train),
                        'y': self.y_train,
                        'ids': self.ids_train,
                        'cols': self.X_train.columns,
                        'group':g_tmp
                    }
                    
                else:
                    self.train = {
                        'X': csr_matrix(self.X_train),
                        'y': self.y_train,
                        'ids': self.ids_train,
                        'cols': self.X_train.columns
                    }
                for x in ['X_train', 'y_train', 'ids_train']: 
                    delattr(self, x) 
                    
            else:
                print('X_train cannot have NULLs')
        else: 
            print('X_train was either ignored or not found')
        
        if (hasattr(self, 'X_val')) & ('X_val' in sets):
            if self.__has_nas(self.X_val)==False:
                
                if group_var_name is not None:
                    g_tmp = self.X_val[group_var_name].values
                    self.X_val = self.X_val.drop(columns=group_var_name)
                    self.val = {
                        'X': csr_matrix(self.X_val),
                        'y': self.y_val,
                        'ids': self.ids_val,
                        'cols': self.X_val.columns,
                        'group':g_tmp
                    }
                    
                else:
                    self.val = {
                        'X': csr_matrix(self.X_val),
                        'y': self.y_val,
                        'ids': self.ids_val,
                        'cols': self.X_val.columns
                    }
                    
                for x in ['X_val', 'y_val', 'ids_val']: 
                    delattr(self, x) 
                    
            else:
                print('X_val cannot have NULLs')
        else: 
            print('X_val was either ignored or not found')
            
        if (hasattr(self, 'X_test')) & ('X_test' in sets):
            if self.__has_nas(self.X_test)==False:
                
                if group_var_name is not None:
                    g_tmp = self.X_test[group_var_name].values
                    self.X_test = self.X_test.drop(columns=group_var_name)
                    self.test = {
                        'X': csr_matrix(self.X_test),
                        'y': self.y_test,
                        'ids': self.ids_test,
                        'cols': self.X_test.columns,
                        'group':g_tmp
                    }
                    
                else:
                    self.test = {
                        'X': csr_matrix(self.X_test),
                        'y': self.y_test,
                        'ids': self.ids_test,
                        'cols': self.X_test.columns
                    }
                for x in ['X_test', 'y_test', 'ids_test']: 
                    delattr(self, x) 
                    
            else:
                print('X_test cannot have NULLs')
        else: 
            print('X_test was either ignored or not found')
        
        return self
    
    def to_torch(
        self, 
        group_var_name = None, 
        balance_groups=False
    ):
        """
        Generates torch datasets using ArrayLoaderGenerator from utils_prediction.nn.datasets.
        The output dataset can be trained by model trainers in the utils_prediction.nn module. 
        This requires all 3 data partitions (train, val, test) to be preprocessed.
        """
        # convert to sparse matrices
        if not hasattr(self,'train') and not hasattr(self,'test'):
            self.to_sparse(group_var_name = group_var_name)
        
        ## get data.cohort
        if group_var_name is not None:
            cols = ['y','ids','group']
        else:
            cols = ['y','ids']
        
        df_train = pd.DataFrame(
            np.array([self.train[x] for x in self.train.keys() if x in cols]
            ).T,columns=cols)
        df_train['fold_id'] = 'train'

        df_val = pd.DataFrame(
            np.array([self.val[x] for x in self.val.keys() if x in cols]
            ).T,columns=cols)
        df_val['fold_id'] = 'val'

        df_test = pd.DataFrame(
            np.array([self.test[x] for x in self.test.keys() if x in cols]
            ).T,columns=cols)
        df_test['fold_id'] = 'test'

        cohort = pd.concat((
            df_train,
            df_val,df_test
        ), ignore_index=True).reset_index().rename(
            columns={'index':'row_id'}
        )

        ## stack features
        features = vstack([
            self.train['X'],
            self.val['X'],
            self.test['X']
        ])

        ## use nn dataloader
        if group_var_name is not None:
            self.loader_generator = ArrayLoaderGenerator(
                features=features,
                cohort=cohort,
                row_id_col="row_id",
                fold_id_test="test",
                label_col="y",
                fold_id="val",
                num_workers=0,
                ids_var = 'ids',
                group_var_name = 'group',
                include_group_in_dataset=True,
                balance_groups = balance_groups
            )
        else:
            self.loader_generator = ArrayLoaderGenerator(
                features=features,
                cohort=cohort,
                row_id_col="row_id",
                fold_id_test="test",
                label_col="y",
                fold_id="val",
                num_workers=0,
                ids_var = 'ids'
            )
        return self.loader_generator.init_loaders()
    
    def to_df(self):
        """
        This will turn all train, val, and test dictionaries (that contain 
        sparse data) in into dataframes like the following:
        
        data.X_train: df containing training features
        data.y_train: array of labels
        data.ids_train: array of row identifiers (subject_id)
        """
        if hasattr(self, 'train'):
            
            self.X_train = pd.DataFrame(
                data = self.train['X'].toarray(), 
                columns = self.train['cols']
            )
            
            self.y_train = self.train['y']
            self.ids_train = self.train['ids']
            delattr(self, 'train')
            
        else:
            print('missing train dict with sparse feature array')
        
        if hasattr(self, 'val'):
            
            self.X_val = pd.DataFrame(
                data = self.val['X'].toarray(), 
                columns = self.val['cols']
            )
            
            self.y_val = self.val['y']
            self.ids_val = self.val['ids']
            delattr(self, 'val')
            
        else: 
            print('missing val dict with sparse feature array')
            
        if hasattr(self, 'test'):
            
            self.X_test = pd.DataFrame(
                data = self.test['X'].toarray(), 
                columns = self.test['cols']
            )
            
            self.y_test = self.test['y']
            self.ids_test = self.test['ids']
            delattr(self, 'test')
            
        else: 
            print('missing test dict with sparse matrix')
        
        return self
    
    def remove_attr(self,attr_list=[]):
        """
        remove attributes from dataloader specified in the attr_list
        should only do this to save memory (e.g., remove features attribute 
        after data has been split)
        """
        for x in attr_list: 
            if hasattr(self,x): delattr(self,x)
    
    def save_datasets(self,fname='default',ftype=None):
        """
        Save datasets in either parquet [default] or feather format.
        """
        if not ftype: 
            ftype = self.config['datasets_ftype']
        
        # Check if data are in dictionaries (sparse matrices)
        if (
            hasattr(self,'train') or 
            hasattr(self,'val') or 
            hasattr(self,'test')
        ): 
            self.to_df()
        
        savepath = os.path.join(
            self.config['datasets_fpath'],
            f"analysis_id={self.config['analysis_id']}",
            "datasets",
            fname
        )
        
        os.makedirs(savepath, exist_ok = True)
        
        # Save datasets
        for element in [
            'X_train',
            'y_train',
            'ids_train',
            'X_val',
            'y_val',
            'ids_val',
            'X_test',
            'y_test',
            'ids_test'
        ]:
            
            if hasattr(self,element): 
                    
                if ftype == 'parquet':
                    pq.write_table(
                        pa.Table.from_pandas(
                            pd.DataFrame(getattr(self,element))
                        ),
                        f"{savepath}/{element}.parquet"
                        ) 
                    
                if ftype == 'feather':
                    feather.write_feather(
                        pd.DataFrame(getattr(self,element)),
                        f"{savepath}/{element}.feather"
                        )
    
    def load_datasets(self,fname='default',ftype=None):
        """
        Load datasets (parquet [default] or feather) into dataloader from hard disk
        """
        if not ftype: 
            ftype = self.config['datasets_ftype']
        
        # Get filenames
        fileslist = [
            'X_train',
            'y_train',
            'ids_train',
            'X_val',
            'y_val',
            'ids_val',
            'X_test',
            'y_test',
            'ids_test'
        ]
        
        filespath = os.path.join(
            self.config['datasets_fpath'],
            f"analysis_id={self.config['analysis_id']}",
            'datasets',
            fname
        )

        filesnames = [
            x for x in os.listdir(filespath) 
            if x.endswith(f".{ftype}") and 
            x.split('.')[0] in fileslist
        ]
        
        # Load files
        for ifile in filesnames:
                
            if ifile.split('.')[0] in ['X_train','X_val','X_test']:
                
                if ftype == 'parquet':
                    setattr(
                        self,
                        ifile.split('.')[0],
                        pq.read_table(f"{filespath}/{ifile}").to_pandas()
                    )
                    
                if ftype == 'feather':
                    setattr(
                        self,
                        ifile.split('.')[0],
                        feather.read_feather(f"{filespath}/{ifile}")
                    )
                    
            else:
                
                if ftype == 'parquet':
                    setattr(
                        self,
                        ifile.split('.')[0],
                        pq.read_table(
                            f"{filespath}/{ifile}"
                        ).to_pandas().to_numpy().ravel()
                    )
                    
                if ftype == 'feather':
                    setattr(
                        self,
                        ifile.split('.')[0],
                        feather.read_feather(
                            f"{filespath}/{ifile}"
                        ).to_numpy().ravel()
                    )
        
        return self

    def __has_nas(self, df):
        return np.sum(np.sum(df.isna()))>0
    
    def __build_default_config(self):
        """
        default config
        """
        return {'analysis_id':'mortality',
            'features_fpath':'/hpf/projects/lsung/projects/mimic4ds/data/',
            'features_ftype':'parquet',
            'datasets_fpath':'/hpf/projects/lsung/projects/mimic4ds/artifacts/',
            'datasets_ftype':'parquet',
            'verbose':False,
            'label_col':'label',
            'id_col':'subject_id'
             }
    
    def __update_config(self, **override_dict):
        """
        update config dict with inputs
        """
        return {**self.config, **override_dict}
    
    def __split_stratified(
        self,
        p_splits=[0.7,0.15,0.15], 
        seed = 44, 
        remove_original = True, 
        retain_group_var = False
    ):
        """
        conduct startified splitting on data based on p_splits.
        If p_splits contains 2 values, data will split into train & test sets.
        If p_splits contains 3 values, data will split into train, val, and test sets.
        
        input: p_splits [train proportion, validation proportion, test proportion]
        """
        if hasattr(self, "features") == False:
            raise Exception("load_features() first")
        
        # split into train, val, test sets 
        if len(p_splits)==3:
            if self.config['verbose']:
                print('Splitting data into train, validation, and test sets')
                print('random seed:', seed)

            ## Create stratified splits
            sss1 = StratifiedShuffleSplit(
                1, 
                test_size=np.sum(p_splits[1:]), 
                random_state=seed
            )
            
            sss2 = StratifiedShuffleSplit(
                1, 
                test_size=p_splits[1]/np.sum(p_splits[1:]), 
                random_state=seed
            )

            y = self.features['label'].values
            train_idx, val_test_idx = next(sss1.split(y, y))

            y_val_test = y[val_test_idx]
            val_idx, test_idx = next(sss2.split(y_val_test, y_val_test))

            val_idx = val_test_idx[val_idx]
            test_idx = val_test_idx[test_idx]

            # Assign labels
            self.y_train, self.y_val, self.y_test = y[train_idx], y[val_idx], y[test_idx]
            
            if self.config['verbose']: 
                print(
                    'Patient splits:', 
                    len(self.y_train), 
                    len(self.y_val), 
                    len(self.y_test)
                )

            # Assign features
            if not retain_group_var:
                
                self.X_train = self.features.loc[train_idx].drop(
                    columns=['label','group_var']
                ).reset_index(drop=True)
                
                self.X_val = self.features.loc[val_idx].drop(
                    columns=['label','group_var']
                ).reset_index(drop=True)
                
                self.X_test = self.features.loc[test_idx].drop(
                    columns=['label','group_var']
                ).reset_index(drop=True)
                
            else:
                self.X_train = self.features.loc[train_idx].drop(
                    columns=['label']
                ).reset_index(drop=True)
                
                self.X_val = self.features.loc[val_idx].drop(
                    columns=['label']
                ).reset_index(drop=True)
                
                self.X_test = self.features.loc[test_idx].drop(
                    columns=['label']
                ).reset_index(drop=True)

            # Save IDs
            self.ids_train = self.X_train[self.config['id_col']].values
            self.ids_val = self.X_val[self.config['id_col']].values
            self.ids_test = self.X_test[self.config['id_col']].values

            # drop IDs
            self.X_train.drop(
                columns=[self.config['id_col']],
                inplace=True
            )
            
            self.X_val.drop(
                columns=[self.config['id_col']],
                inplace=True
            )
            
            self.X_test.drop(
                columns=[self.config['id_col']],
                inplace=True
            )
        
        if len(p_splits)==2:

            ## Create stratified splits
            sss1 = StratifiedShuffleSplit(
                1, test_size=p_splits[1], random_state=seed
            )

            y = self.features['label'].values
            train_idx, test_idx = next(sss1.split(y, y))

            # Assign labels
            self.y_train, self.y_test = y[train_idx], y[test_idx]

            if self.config['verbose']: 
                print(
                    'Patient splits:', 
                    len(self.y_train), 
                    len(self.y_test)
                )

            # Assign features
            if not retain_group_var:
                self.X_train = self.features.loc[train_idx].drop(
                    columns=['label','group_var']
                ).reset_index(drop=True)
                
                self.X_test = self.features.loc[test_idx].drop(
                    columns=['label','group_var']
                ).reset_index(drop=True)
                
            else:
                self.X_train = self.features.loc[train_idx].drop(
                    columns=['label']
                ).reset_index(drop=True)
                
                self.X_test = self.features.loc[test_idx].drop(
                    columns=['label']
                ).reset_index(drop=True)

            # Save IDs
            self.ids_train = self.X_train[self.config['id_col']].values
            self.ids_test = self.X_test[self.config['id_col']].values

            # drop IDs
            self.X_train.drop(
                columns=[self.config['id_col']],
                inplace=True
            )
            
            self.X_test.drop(
                columns=[self.config['id_col']],
                inplace=True
            )
            
        if remove_original:
            self.remove_attr(['features'])
        
        return self