"""
Featurizer that extracts features from MIMIC-IV stored on GBQ
"""
import os
import pickle

import pandas as pd
import numpy as np 
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.feather as feather

from google.cloud import bigquery
from google.oauth2 import service_account

class DischargeCountQuery():
    """
    Outputs a query that counts occurences of concept_ids across all
    discharge datetimes within the specified time bin.
    This ensures that only previous admissions are counted, NOT the
    current admision. 
    
    Input: 
        A dictionary with the following keys:
            project_name:   project name on GBQ (e.g., mimic-iv-ches)
            cohort_table:   the cohort table of interest
            concept_table:  the concept table of interest
            concept_id:     the codes to be counted (e.g., icd_code)
            bin_left:       left boundary for time bin. If we're looking at 0-7 days before ICU admission, left bin should be -7, and right should be 0.
            bin_right:      right boundary for time bin
    Output:
        A GBQ query that can be used with gbq query functions from utils_db.
    
    Usage Example:
        config = {'project_name':'mimic-iv-ches',
                 'cohort_table':'cohorts.mimic4ds_inhospmort',
                 'concept_table':'hosp.diagnoses_icd',
                 'concept_id':'icd_code',
                 'bin_left': '-7',
                 'bin_right': '0'}
        q = DischargeCountQuery(config).get_base_query()
        df = gbq_query(c,q) # returns dataframe that contains the query output
    """
    
    def __init__(self, config, **kwargs):
        self.__dict__.update(config)
    
    def get_base_query(self):
        q = f"""
            with source_table as (
                select
                    t1.subject_id, t1.stay_id, t1.ref_datetime,
                    -- ref_datetime and ref_date for each patient have 1-to-1 mapping because
                    -- we have one unique stay_id per patient
                    cast(t1.ref_datetime as date) as ref_date, 
                    t2.hadm_id,
                    cast(t3.{self.concept_id} as string) as concept_id,
                    concat('bin_',{self.bin_left},'_',{self.bin_right}) as time_bin
                from
                    `{self.project_name}.{self.cohort_table}` t1
                    inner join `{self.project_name}.core.admissions` t2 on t1.subject_id = t2.subject_id
                    inner join `{self.project_name}.{self.concept_table}` t3 on t2.hadm_id = t3.hadm_id
                where
                    t2.dischtime between
                        datetime_add(t1.ref_datetime, interval {self.bin_left} day) and
                        datetime_add(t1.ref_datetime, interval {self.bin_right} day)
            )
            select
                subject_id, concept_id, time_bin, count(distinct hadm_id) as n_admissions,
                ref_date,
                concat(concept_id,'_{self.feature_tag}','_{self.feature_type}_',time_bin) as feature_id,
                count(*) as feature_val
            from source_table
            group by subject_id, ref_date, concept_id, time_bin
        """
        return q    
    
    
class CharttimeCountQuery():
    """
    Outputs a query that counts occurences of concept_ids across all datetimes
    that occured within the specified timebin. This query should be used for
    concepts that are charted "real-time", e.g., itemID / LOINC for labevents
    that have charttime.
    
    Input: 
        A dictionary with the following keys:
            project_name:   project name on GBQ (e.g., mimic-iv-ches)
            cohort_table:   the cohort table of interest
            concept_table:  the concept table of interest
            cocnept_id:     the codes to be counted (e.g., NDC)
            bin_left:       left boundary for time bin. If we're looking at 0-7 days before ICU admission, left bin should be -7, and right should be 0.
            bin_right:      right boundary for time bin
            timestamp:      the datetime stamp used to determine whether the code should be counted (e.g., charttime for labevents)
    Output:
        A GBQ query that can be used with gbq query functions from utils_db.
    
    Usage Example:
        config = {'project_name':'mimic-iv-ches',
                 'cohort_table':'cohorts.mimic4ds_inhospmort',
                 'concept_table':'hosp.labevents',
                 'concept_id':'itemid',
                 'bin_left': '-7',
                 'bin_right': '0',
                 'timestamp': 'charttime'}
        q = CharttimeCountQuery(config).get_base_query()
        df = gbq_query(c,q)
    """
    def __init__(self, config, **kwargs):
        self.__dict__.update(config)
    
    def get_base_query(self):
        q = f"""
            with source_table as (
                select
                    t1.subject_id, t1.stay_id, t1.ref_datetime,
                    -- ref_datetime and ref_date for each patient have 1-to-1 mapping because
                    -- we have one unique stay_id per patient.
                    cast(t1.ref_datetime as date) as ref_date,
                    t2.hadm_id,
                    cast(t3.{self.concept_id} as string) as concept_id,
                    concat('bin_',{self.bin_left},'_',{self.bin_right}) as time_bin
                from
                    `{self.project_name}.{self.cohort_table}` t1
                    inner join `{self.project_name}.core.admissions` t2 on t1.subject_id = t2.subject_id
                    inner join `{self.project_name}.{self.concept_table}` t3 on t2.hadm_id = t3.hadm_id
                where
                    t3.{self.timestamp} between
                        datetime_add(t1.ref_datetime, interval cast({self.bin_left}*24 as int64) hour) and
                        datetime_add(t1.ref_datetime, interval cast({self.bin_right}*24 as int64) hour)
            )
            select
                subject_id, concept_id, time_bin, count(distinct hadm_id) as n_admissions,
                ref_date,
                concat(concept_id,'_{self.feature_tag}','_{self.feature_type}_',time_bin) as feature_id,
                count(*) as feature_val
            from source_table
            group by subject_id, ref_date, concept_id, time_bin
        """
        return q 

    
class CharttimeMeasurementQuery():
    """
    Outputs a query that calculates summary statistics (min,max,avg) of concepts across all 
    datetimes that occured within the specified timebin. This query should be 
    used for concepts that are charted "real-time", e.g., chartevents or labevents.
    
    Input: 
        A dictionary with the following keys:
            project_name:   project name on GBQ (e.g., mimic-iv-ches)
            cohort_table:   the cohort table of interest
            concept_table:  the concept table of interest (e.g., labevents)
            cocnept_id:     the codes to be counted (e.g., NDC)
            bin_left:       left boundary for time bin. If we're looking at 0-7 days before ICU admission, left bin should be -7, and right should be 0.
            bin_right:      right boundary for time bin
            timestamp:      the datetime stamp used to determine whether the code should be counted (e.g., charttime for labevents)
    Output:
        A GBQ query that can be used with gbq query functions from utils_db.
    
    Usage Example:
        config = {'project_name':'mimic-iv-ches',
             'cohort_table':'cohorts.mimic4ds_inhospmort',
             'concept_table':'hosp.labevents',
             'concept_id':'itemid',
             'bin_left': '-7',
             'bin_right': '0',
             'timestamp': 'charttime'}
        q = CharttimeMeasurementQuery(config).get_base_query()
        df = gbq_query(c,q)
    """
    def __init__(self, config, **kwargs):
        self.__dict__.update(config)
        self.stats = ['min','max','avg']
    
    def measurement_query(self,stat):
        '''
        build query that extracts a specific stat from the concept table
        '''
        q = f"""
            with source_table as (
                select
                    t1.subject_id, t1.stay_id, t1.ref_datetime,
                    -- ref_datetime and ref_date for each patient have 1-to-1 mapping because
                    -- we have one unique stay_id per patient.
                    cast(t1.ref_datetime as date) as ref_date,
                    t2.hadm_id,
                    cast(t3.{self.concept_id} as string) as concept_id,
                    t3.valuenum, t3.value,
                    concat('bin_',{self.bin_left},'_',{self.bin_right}) as time_bin
                from
                    `{self.project_name}.{self.cohort_table}` t1
                    inner join `{self.project_name}.core.admissions` t2 on t1.subject_id = t2.subject_id
                    inner join `{self.project_name}.{self.concept_table}` t3 on t2.hadm_id = t3.hadm_id
                where
                    t3.{self.timestamp} between
                        datetime_add(t1.ref_datetime, interval cast({self.bin_left}*24 as int64) hour) and
                        datetime_add(t1.ref_datetime, interval cast({self.bin_right}*24 as int64) hour)
            )
            select
                subject_id, concept_id, time_bin, count(distinct hadm_id) as n_admissions,
                ref_date,
                concat(concept_id,'_','{self.feature_tag}_{self.feature_type}_','{stat}','_',time_bin) as feature_id,
                {stat}(coalesce(valuenum,safe_cast(value as FLOAT64))) as feature_val
            from source_table
            where valuenum is not null or value is not null
            group by subject_id, ref_date, concept_id, time_bin
        """
        return q 
    
    def get_base_query(self):
        '''
        stitch individual stat queries together using union distinct
        '''
        for i,stat in enumerate(self.stats):
            if i == 0:
                q = '('+self.measurement_query(stat)+')\n'
            else:
                q+= 'union distinct ('+self.measurement_query(stat)+')\n'
        return q
        
        
class DemographicQuery():
    """
    Outputs a query that obtains demographic variables for all subjects
    in the cohort. 
    
    This includes the label column from the cohort table!
    
    Note that the output for Demographic Query is in a different format. 
    Whereas measurement and counts have feature_id and feature_values, 
    demographic_vars have features as separate columns. This is due to
    the demographic_vars having Varchar datatype that can't be concatenated
    with numerical data types.. unless they're one-hot encoded. But 
    the complexity that comes with it and the need to format it back into 
    column formats makes it more convenient to just concatenate (axis=1) 
    with the rest of the variables when they're transformed back into column
    formats.
    
    Extracted demographic variables: 
        Patients Table: gender
        Admissions Table: insurance, language, marital_status, ethnicity, age (year from admittime - anchor_year + anchor_age)
    
    Input: 
        A dictionary with the following keys:
            project_name:   project name on GBQ (e.g., mimic-iv-ches)
            cohort_table:   the cohort table of interest
    Output:
        A GBQ query that can be used with gbq query functions from utils_db.
    
    Usage Example:
        config = {'project_name':'mimic-iv-ches',
                 'cohort_table':'cohorts.mimic4ds_inhospmort'}
        q = DemographicQuery(config).get_base_query()
        df = gbq_query(c,q)
    """
    def __init__(self, config, **kwargs):
        self.__dict__.update(config)
        
    def get_base_query(self):
        q = f"""
            with source_table as (
                select
                    t1.label,
                    t1.group_var,
                    t1.subject_id, 
                    t2.insurance, t2.marital_status, t2.ethnicity,
                    case when t2.language not in ('ENGLISH') then 'OTHER' else t2.language end as language,
                    extract(year from t2.admittime) - t3.anchor_year + t3.anchor_age as age,
                    t3.gender,
                from
                    `{self.project_name}.{self.cohort_table}` t1
                    inner join `{self.project_name}.core.admissions` t2 on t1.hadm_id = t2.hadm_id
                    inner join `{self.project_name}.core.patients` t3 on t1.subject_id = t3.subject_id
            )
            select *
            from source_table
        """
        return q 
    
class featurizer():
    """
    stitches queries and extract from GBQ
    """
    def __init__(self,config={},**kwargs):
        # Initialize output_dict that contains:
        #     1. featurizer input config
        #     2. query configs
        #     3. queries for time_dependent and time_invariant features
        self.output_dict = {}
        self.output_dict['query_configs'] = []
        
        # update featurizer params using config
        if not config: 
            config = self.build_default_config() 
            
        self.output_dict['featurizer_config'] = config
        self.__dict__.update(config)
        
        # GBQ connect
        self.gbq_connect() 
        
        if self.include_all_history is True: 
            self.time_bins = ['-365000'] + self.time_bins
            
    def gbq_connect(
        self,
        service_account_json_path = '/hpf/projects/lsung/projects/mimic4proj/access/gcp.json', # path to your key file
        project_id = None
    ):
        """
        Establishes connection with GBQ using service account file and project_id.
        """
        if project_id is None:
            project_id = self.project_name
        
        cred = service_account.Credentials.from_service_account_file(service_account_json_path)
        project_id = project_id
        
        self.c = bigquery.Client(
            credentials = cred, 
            project = project_id
        )
        
        print('Google Big Query Connection Established')
        
        return self
    
    def build_default_config(self):
        """
        Get default configuration dictionary
        """
        return {
            'analysis_id':'mortality_withICU',
            'project_name':'mimic-iv-ches',
            'cohort_table':'cohorts.mimic4ds_inhospmort',
            'tables_to_build':{
                'count_labs':{
                    'feature_tag':'lab',
                    'feature_type':'count',
                    'concept_table':'hosp.labevents',
                    'concept_id':'itemid',
                    'timestamp':'charttime'
                },
                'count_prescriptions':{
                    'feature_tag':'presc',
                    'feature_type':'count',
                    'concept_table':'hosp.prescriptions',
                    'concept_id':'ndc',
                    'timestamp':'starttime'
                },
                'count_diagnosis':{
                    'feature_tag':'diag',
                    'feature_type':'count',
                    'concept_table':'hosp.diagnoses_icd',
                    'concept_id':'icd_code',
                    'timestamp': 'discharge'
                },
                'count_procedures':{
                    'feature_tag':'proc',
                    'feature_type':'count',
                    'concept_table':'hosp.procedures_icd',
                    'concept_id':'icd_code',
                    'timestamp': 'discharge'
                },
                'count_hcpcs':{
                    'feature_tag':'hcpcs',
                    'feature_type':'count',
                    'concept_table':'hosp.hcpcsevents',
                    'concept_id':'hcpcs_cd',
                    'timestamp': 'discharge'
                },
                'meas_labs':{
                    'feature_tag':'labs',
                    'feature_type':'measurement',
                    'concept_table':'hosp.labevents',
                    'concept_id':'itemid',
                    'timestamp':'charttime'
                },
                'meas_icucharts':{
                    'feature_tag':'icucharts',
                    'feature_type':'measurement',
                    'concept_table':'icu.chartevents',
                    'concept_id':'itemid',
                    'timestamp':'charttime'
                },
            },
            'time_bins': ['-180','-30','-7','0','0.167'],
            'include_all_history': True,
            'n_jobs':4, # Currently doesn't add anything
            'save_fpath':'/hpf/projects/lsung/projects/mimic4ds/data/',
            'save_ftype':'feather'
         }
    
    def build_query_config(self):
        
        return {
            'project_name':self.project_name,
            'cohort_table':self.cohort_table,
            'concept_table':self.concept_table,
            'concept_id':self.concept_id,
            'feature_tag':self.feature_tag,
            'feature_type':self.feature_type,
            'timestamp':self.timestamp,
            'bin_left':self.bin_left,
            'bin_right':self.bin_right
        }
    
    def build_time_bins(self):
        """
        turn time_bins into bins_left and bins_right
        """
        self.bins_left, self.bins_right = [],[]
        
        for i in range(len(self.time_bins)-1):
            self.bins_left.append(self.time_bins[i])
            self.bins_right.append(self.time_bins[i+1])
        
        return self
    
    def stitch_timedepfeatures(self):
        """
        build a single query that combines (union all) all the base queries for time-dependent features together
        """
        self.build_time_bins()
        for idxtimebin in range(len(self.bins_left)):
            self.bin_left = self.bins_left[idxtimebin]
            self.bin_right = self.bins_right[idxtimebin]
            for n,tbl_config in enumerate(self.tables_to_build):
                # it's cleaner to build a new dictionary with the keys instead of updating self.__dict__, will implement later
                self.__dict__.update(self.tables_to_build[tbl_config])
                config = self.build_query_config()
                
                # for count at discharge, do not include bin-cuttoffs past ref_datetime (e.g., ICU admission date time)
                if (
                    (self.feature_type == 'count') & 
                    (self.timestamp == 'discharge') & 
                    (float(self.bin_right) <=0)
                ):
                    q = DischargeCountQuery(config).get_base_query()
                    self.output_dict['query_configs'].append(config) 
                
                elif (
                    (self.feature_type == 'count') & 
                    ((self.timestamp == 'starttime') | (self.timestamp == 'charttime'))
                ):
                    q = CharttimeCountQuery(config).get_base_query()
                    self.output_dict['query_configs'].append(config) 
                
                # if ICU in feature_tag, grab only those with timestamps after ICU admission but before cutoff time.
                elif (
                    (self.feature_type == 'measurement') & 
                    ((self.timestamp == 'starttime') | (self.timestamp == 'charttime')) & 
                    ('icu' in self.feature_tag) & 
                    (float(self.bin_left)>=0)
                ):
                    q = CharttimeMeasurementQuery(config).get_base_query()
                    self.output_dict['query_configs'].append(config) 
                
                # if ICU is not in feature_tag, grab all past measurements and measurements after ICU admission 
                # but before cutoff time
                elif (
                    (self.feature_type == 'measurement') & 
                    ((self.timestamp == 'starttime') | (self.timestamp == 'charttime')) &
                    ('icu' not in self.feature_tag)
                ):
                    q = CharttimeMeasurementQuery(config).get_base_query()
                    self.output_dict['query_configs'].append(config) 
                
                # Stitch queries together
                if idxtimebin == 0 and n == 0:
                    self.query = '(' + q + ')\n'
                else:
                    self.query += 'union distinct ('+ q +')\n'
        
        return self

    def preproc_timedepfeatures(self):
        """
        preprocess the stitched query:
            1. Remove NaN in feature_val column
            2. Load into dataframe and run pivot_table to convert features into columns
            3. Turn NaNs in count columns to 0
        """
        self.stitch_timedepfeatures()
        
        self.query = f"""
            select subject_id, feature_id, feature_val
            from ({self.query})
            where feature_val is not null
            """
        
        # save time-dependent query
        self.output_dict['query_time_dependent'] = self.query
        
        # query and save to df
        results = self.c.query(self.query)
        df = results.to_dataframe()
        
        # pivot table so features become columns
        df = df.pivot_table(
            index='subject_id',
            columns='feature_id'
            ,values='feature_val'
        )
        
        # set NaN count values to 0
        cols_count = [
            x for x in df.columns if 'count' in x
        ]
        
        df[cols_count] = df[cols_count].fillna(0)
        
        return df
    
    def featurize(self):
        """
        Extract demographic variable for each subject in the cohort and join with 
        time-dependent features
        """
        # run time-dependent query
        df_time = self.preproc_timedepfeatures()
        
        # build time-invariant query
        config = self.build_query_config()
        q = DemographicQuery(config).get_base_query()
        
        # save time-invariant query
        self.output_dict['query_time_invariant'] = q
        
        # query time-invariant features and save to df
        results = self.c.query(q)
        df_demographics = results.to_dataframe()
        
        # merge df and return
        return df_demographics.merge(
            df_time, 
            on = 'subject_id', 
            how='left'
        )
        
    def featurize_and_save(self):
        """
        save features as parquet file 
        """
        
        df = self.featurize()
        
        # Create path if does not exist
        dir_path = os.path.dirname(self.save_fpath+f'analysis_id={self.analysis_id}/')
        os.makedirs(dir_path, exist_ok=True)
        
        # Save to file
        print('saving features...')
        if self.save_ftype == 'parquet':
            pq.write_table(
                pa.Table.from_pandas(df),
                self.save_fpath+f'analysis_id={self.analysis_id}/features.parquet'
            )
            
        elif self.save_ftype == 'feather':
            feather.write_feather(
                df,
                self.save_fpath+f'analysis_id={self.analysis_id}/features.feather'
            )
        
        # Save self.output_dict as pickle
        with open(self.save_fpath+f'analysis_id={self.analysis_id}/featurizer_configs.pkl', 'wb') as f:
            
            pickle.dump(self.output_dict, f)
            
        return df