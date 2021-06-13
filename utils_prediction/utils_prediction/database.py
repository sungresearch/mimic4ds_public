"""
Helper functions to connect to and query from databases

Database supports:
- Google Bigquery (gbq)
"""

from google.cloud import bigquery
from google.oauth2 import service_account
from time import time

## Connect to GBQ
def gbq_connect(
    service_account_json_path = '/hpf/projects/lsung/creds/gbq/mimic.json',
    project_id = 'mimic-iv-ches'
    ):
    
    cred = service_account.Credentials.from_service_account_file(service_account_json_path)
    project_id = project_id
    c = bigquery.Client(credentials = cred, project = project_id)
    print('Google Big Query Connection Established')
    return c

def gbq_query(
    c,
    q,
    verbose=True
    ):
    
    '''
    Executes query (q) on GBQ using the client (c)
    c = gbq_connect()
    q = query in string format. E.g.,:
        'select * from tbl limit 10'
    '''
    t = time()
    results = c.query(q)
    df = results.to_dataframe()
    if verbose:
        print(
            '\n',
            'Done!\n',
            'Took',round(time()-t,2),'s to process the query.\n',
            'Your query returned',len(df),'rows,',len(df.columns),'columns.\n',
            'Total Memory Usage:',round(df.memory_usage(index=True).sum()/1000000,2),'MB.\n\n'
         )
        
    return df

def gbq_query_w_file(
    c,
    sql_path,
    verbose=True
    ):
    '''
    Execute query script located at sql_path using the client (c)
    c = gbq_connect()
    '''
    t = time()
    
    with open(sql_path,'r') as sql_file:
        q = sql_file.read()
    
    results = c.query(q)
    df = results.to_dataframe()
    
    if verbose:
        print(
            '\n',
            'Done!\n',
            'Took',round(time()-t,2),'s to process the query.\n',
            'Your query returned',len(df),'rows,',len(df.columns),'columns.\n',
            'Total Memory Usage:',df.memory_usage(index=True).sum()/1000000,'MB.\n\n'
        )
        
    return df

