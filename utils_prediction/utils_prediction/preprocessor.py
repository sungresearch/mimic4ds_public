"""
Preprocessors compatible with scikit-learn
"""

import numpy as np
import pandas as pd 
from sklearn.preprocessing import OneHotEncoder

class prune_features():
    """
    Remove features that have less than n subjects (thresh) worth of data
    
    Can consider special cases too, e.g., if 0 is considered missing data for
    some columns, user can specify a tag for those special columns. Otherwise, 
    only NaNs are considered as missing data.
    
    Example special column specifier where count is the tag for the special
    columns and 0s are considered as missing values for those columns:
    
    special_cols = {'count':0}
    
    for now only 1 tag can be used.
    """
    def __init__(
        self, 
        thresh = 25, 
        special_cols = {}
        ):
        
        self.thresh = thresh
        self.special_cols = special_cols
        self.cols_to_keep = []
        self.cols_to_remove = []
        
    def fit(self, df, y = None):
        """
        given df, find columns with at least n subjects worth of data (thresh)
        """
        if self.special_cols:
            # extract columns with tag from special_cols
            spec_cols = [
                x for x in df.columns 
                if next(iter(self.special_cols.keys())) in x
            ]
            
            spec_val = self.special_cols[next(iter(self.special_cols.keys()))]
            
            cols = [
                x for x in df.columns if x not in spec_cols
            ]
   
            # find cols with at least thresh
            c = np.sum(df[cols].notnull())>self.thresh
            self.cols_to_keep += c[c==True].index.values.tolist()
            
            # find cols in special_cols with at least thresh 
            c2 = np.sum(
                (df[spec_cols]!=spec_val) & (df[spec_cols].notnull())
            )>self.thresh
            
            self.cols_to_keep += c2[c2==True].index.values.tolist()
            
        else:
            cols = df.columns
            
            c = np.sum(
                df[cols].notnull()
            )>self.thresh
            
            self.cols_to_keep += c[c==True].index.values.tolist()
            
        self.cols_to_remove = [x for x in df.columns if x not in self.cols_to_keep]
        return self
    
    def transform(self, df):
        """
        given df, return df with only columns to keep
        """
        check_cols = [
            x for x in self.cols_to_keep if x not in df.columns
        ]
        
        if len(check_cols) > 0:
            print(
                len(check_cols),
                'columns not found, creating columns of 0s in the input dataframe'
            )
            
            df.loc[:,check_cols] = 0
            
        return df[self.cols_to_keep]
    
    def fit_transform(self, df, y = None):
        """
        fit, then transform df
        """
        self.fit(df)
        return self.transform(df)
    
    def get_params(self):
        """
        return all parameters in class as a dictionary
        """
        return self.__dict__


class discretizer():
    """
    Discretize features in in dataframe into n_bins.
    
    features with tags in "feature_tags_to_include" are included
    features with tags in "feature_tags_to_exclude" are excluded
    """
    def __init__(
        self, 
        n_bins = 5, 
        feature_tags_to_include = [], 
        feature_tags_to_exclude = []
        ):
        
        self.n_bins = n_bins
        self.feature_tags_to_include = feature_tags_to_include
        self.feature_tags_to_exclude = feature_tags_to_exclude
        self.cols = []
        
    def get_percentiles(self):
        """
        given number of bins (self.n_bins), return a list of percentiles
        
        input: n_bins = 3
        output: [33.3333, 66.6667]
        """
        self.percentiles = np.linspace(0,100,self.n_bins+1)[1:-1].tolist()
        return self
    
    
    def fit(self,df, y = None):
        """
        vectorized percentile computation while ignore NaNs.
        Then convert to dictionary with columns as keys and a list of
        the computed percentiles as values.
        """
        self.get_percentiles()
        
        if (not self.feature_tags_to_include) & (not self.feature_tags_to_exclude):
            self.cols = df.columns
            
        if self.feature_tags_to_include:
            for tag in self.feature_tags_to_include:
                self.cols += [x for x in df.columns if tag in x]
        if self.feature_tags_to_exclude:
            for tag in self.feature_tags_to_exclude:
                self.cols += [x for x in df.columns if tag not in x]
        
        self.cols = list(dict.fromkeys(self.cols))
            
        # Below is a bandaid-solution for monotonicity check error, should look into a better way to do it******
        pct = np.round(np.nanpercentile(df[self.cols], self.percentiles, axis=0),4) 
        
        a = np.full((1,len(self.cols)),-np.inf)
        b = np.full((1,len(self.cols)),np.inf)
        pct = np.concatenate((a,pct,b),axis=0)
        self.bins = pd.DataFrame(data = pct, columns = self.cols).to_dict('list')
        return self
    
    def transform(self,df):
        """
        np.digitize discretizes nan value to the highest bin number +1 (n_bin+1). 
        Here, they're replaced with 0, and so 0 indicate missing value (NaNs)
        
        Need to do: need column checker because the discretizer currently assumes 
        that the df's columns follow the same order and are exactly the same as 
        self.cols. This is not a problem for mimic4ds project because all features
        are extracted together across all years. 
        """
        c_df = df.copy()
        X = c_df[self.cols].values
        for i,col in enumerate(self.cols):
            X[:,i] = np.digitize(X[:,i], self.bins[col])
            
        c_df[self.cols] = X
        c_df[self.cols] = c_df[self.cols].replace(6,0)
        return c_df
    
    def fit_transform(self,df, y = None):
        self.fit(df)
        return self.transform(df)
    
    def get_params(self):
        """
        return all parameters in class as a dictionary
        """
        return self.__dict__
    

class one_hot_encoder():
    """
    fits sklearn's onehotencoder and use it to transform dfs
    
    features with tags in "feature_tags_to_include" are included
    features with tags in "feature_tags_to_exclude" are excluded
    
    can specify output in the form of 'dataframe' (default) or 'sparse' matrix
    ** currently sparse matrix output is not implemented **
    """
    def __init__(
        self, 
        handle_unknown='ignore', 
        output='dataframe', 
        categories = 'auto', 
        drop = None,
        feature_tags_to_include = [], 
        feature_tags_to_exclude = []):
        
        self.m = OneHotEncoder(handle_unknown = handle_unknown, drop=drop, categories = categories)
        self.handle_unknown = handle_unknown
        self.output = output
        self.feature_tags_to_include = feature_tags_to_include
        self.feature_tags_to_exclude = feature_tags_to_exclude
        self.cols = []
        
    def fit(self, df, y = None):
        if (not self.feature_tags_to_include) & (not self.feature_tags_to_exclude):
            self.cols = df.columns
            
        if self.feature_tags_to_include:
            for tag in self.feature_tags_to_include:
                self.cols += [x for x in df.columns if tag in x]
        if self.feature_tags_to_exclude:
            self.cols = [
                x for x in df.columns if not 
                any([True for y in self.feature_tags_to_exclude if y in x])
            ]
            #for tag in self.feature_tags_to_exclude:
                #self.cols += [x for x in df.columns if tag not in x]
        
        self.cols = list(dict.fromkeys(self.cols))
        self.m.fit(df.loc[:,self.cols])
        self.cols_encoded = self.m.get_feature_names(self.cols)
        return self
    
    def transform(self, df):
        other_cols = [x for x in df.columns if x not in self.cols]
        encoded = self.m.transform(df[self.cols])
        if self.output == 'dataframe':
            encoded = pd.DataFrame(data=encoded.toarray(), columns = self.cols_encoded)
            out = pd.concat([encoded,df[other_cols]],axis=1)
        else:
            raise Exception(self.output,'is not implemented')
        return out
    
    def fit_transform(self, df, y = None):
        self.fit(df)
        return self.transform(df)
    
    def get_params(self):
        """
        return all parameters in class as a dictionary
        """
        return self.__dict__
    

class fill_missing():
    """
    fill nan values in df with specified filler
    
    input:
        config = {'tag':filler}
        
        'tag' specifies how to search for columns.
        filler specifies what to replace nans with.
        can have as many tags as needed.
    
    Example:
        config = {'count':0}
    
        for any column with count as part of its name, fill nans with 0.
    """
    def __init__(self, config):
        self.config = config
    
    def fit(self,df,y=None):
        return self
    
    def transform(self, df):
        for i in self.config:
            cols = [x for x in df.columns if i in x]
            df.loc[:,cols] = df[cols].fillna(self.config[i])
        return df

    def fit_transform(self,df,y=None):
        self.fit(df)
        return self.transform(df)
    
    def get_params(self):
        """
        return all parameters in class as a dictionary
        """
        return self.__dict__
    

class binary_discretizer():
    """
    Special binary discretizer that turns non-zero values into 1 and keep 0s as 0s.
    This is used to discretize count features
    """
    
    def __init__(self, feature_tags_to_include = [], feature_tags_to_exclude = []):
        self.feature_tags_to_include = feature_tags_to_include
        self.feature_tags_to_exclude = feature_tags_to_exclude
        self.cols = []
    
    def fit(self,df,y=None):
        # select columns
        if (not self.feature_tags_to_include) & (not self.feature_tags_to_exclude):
            self.cols = df.columns
            
        if self.feature_tags_to_include:
            for tag in self.feature_tags_to_include:
                self.cols += [x for x in df.columns if tag in x]
        if self.feature_tags_to_exclude:
            for tag in self.feature_tags_to_exclude:
                self.cols += [x for x in df.columns if tag not in x]
        
        self.cols = list(dict.fromkeys(self.cols))
            
        return self
    
    def transform(self, df):
        c_df = df.copy()
        m = df[self.cols] == 0
        c_df.loc[:,self.cols] = df[self.cols].where(m,1)
        return c_df
    
    def fit_transform(self, df, y=None):
        self.fit(df)
        return self.transform(df)
    
    def get_params(self):
        """
        return all parameters in class as a dictionary
        """
        return self.__dict__
    