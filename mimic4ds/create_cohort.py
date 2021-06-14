"""
Create cohort using sql files
Available cohorts:
    - mortality (7 day in-hospital mortality)
    - longlos (long length of stay (>3 days))
    - invasivevent (invasive ventilation within 24 hours)
    - sepsis3 (in-icu sepsis onset within 24 hours)
"""
from utils_prediction.database import *
import argparse, os

parser = argparse.ArgumentParser(
    description = "Create Cohort on GBQ using sql files"
    )

parser.add_argument(
    "--sql_fpath",
    type = str,
    default = "/hpf/projects/lsung/projects/public/mimic4ds_public/cohort_sql_files",
    help = "where cohort sql files are stored"
)

parser.add_argument(
    "--cohorts",
    nargs='+',
    default = ['mortality','longlos','invasivevent', 'sepsis3'],
    help = "cohort tables to create on gbq"
)

if __name__ == "__main__":
    
    # init database connector
    c = gbq_connect()
    
    args = vars(parser.parse_args())
    
    for cohort in args['cohorts']:
        fpath = os.path.join(
            args['sql_fpath'],
            f"{cohort}.sql"
        )
        print(f"creating {cohort} cohort table on GBQ")
        _ = gbq_query_w_file(c, fpath, verbose=False)
    
    print('Done!')