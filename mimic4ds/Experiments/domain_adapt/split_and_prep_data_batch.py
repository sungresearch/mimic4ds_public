"""
Generate and submit bash scripts to perform grid search over model hyperparameters
"""

import os

#------------------------------------------------------------------------
# PBS vars
#------------------------------------------------------------------------
script_tag = 'r'

# resources
walltime = "23:00:00"
ppn = "16"
mem = "100g"
vmem = "100g"
node_type = f"nodes=1:ppn={ppn}"

# error & output paths
error_path = "/hpf/projects/lsung/projects/public/mimic4ds_public/mimic4ds/Experiments/domain_adapt/logs/split_and_preprocess/error"
output_path = "/hpf/projects/lsung/projects/public/mimic4ds_public/mimic4ds/Experiments/domain_adapt/logs/split_and_preprocess/output"

if not os.path.exists(error_path): os.makedirs(error_path)
if not os.path.exists(output_path): os.makedirs(output_path)
    
# Job dependencies
# -- If not None, submitted jobs will be held until the {job_id} completes 
# -- without error.
job_id = None


#------------------------------------------------------------------------
# Tasks
#------------------------------------------------------------------------
tasks = ['invasivevent','sepsis','longlos','mortality']
                    
#------------------------------------------------------------------------
# Create and submit jobs
#------------------------------------------------------------------------
c = 0
for task in tasks:
    c+=1
    # create bash script
    with open (f"{script_tag}{c}.sh", 'w') as rsh:
        rsh.write(f'''\
#!/bin/bash
#PBS -N preproc{task[:2].upper()}
#PBS -l walltime={walltime}
#PBS -l {node_type}
#PBS -l mem={mem},vmem={vmem}
#PBS -m be
#PBS -e {error_path}
#PBS -o {output_path}

source activate /hpf/projects/lsung/envs/anaconda/mimic4ds
cd /hpf/projects/lsung/projects/public/mimic4ds_public/mimic4ds/Experiments/domain_adapt/scripts

python split_and_prep_data.py \
--analysis_id="{task}"
    ''')
    # run bash script
    if job_id is not None:
        comm = f"-W depend=afterok:{job_id}"
    else:
        comm = ''

    os.system(f'qsub {comm} {script_tag}{c}.sh')
    # remove bash script
    os.remove(f'{script_tag}{c}.sh')
    
