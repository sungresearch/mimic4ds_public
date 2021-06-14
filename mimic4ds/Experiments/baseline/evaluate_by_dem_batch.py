"""
Generate and submit bash scripts to evaluate trained models
"""

import os

#------------------------------------------------------------------------
# PBS vars
#------------------------------------------------------------------------
script_tag = 'r'

# resources
walltime = "23:00:00"
ppn = "8"
mem = "100g"
vmem = "100g"
node_type = f"nodes=1:ppn={ppn}"
use_gpu = False

# error & output paths
error_path = "/hpf/projects/lsung/projects/public/mimic4ds_public/mimic4ds/Experiments/baseline/logs/evaluate_by_dem/error"
output_path = "/hpf/projects/lsung/projects/public/mimic4ds_public/mimic4ds/Experiments/baseline/logs/evaluate_by_dem/output"

if not os.path.exists(error_path): os.makedirs(error_path)
if not os.path.exists(output_path): os.makedirs(output_path)
    
# Job dependencies
# -- If not None, submitted jobs will be held until the {job_id} completes 
# -- without error.
job_id = None


#------------------------------------------------------------------------
# Tasks
#------------------------------------------------------------------------
tasks = ['longlos','sepsis','mortality','invasivevent']
groups = ["2008 - 2010", "2011 - 2013", "2014 - 2016", "2017 - 2019"]
dem_groups = ['gender']
                    
#------------------------------------------------------------------------
# Create and submit jobs
#------------------------------------------------------------------------
c = 0
for group in dem_groups:
    for task in tasks:
        for g,train_group in enumerate(groups):
            for eval_group in groups[g:]:
                c+=1
                # create bash script
                with open (f"{script_tag}{c}.sh", 'w') as rsh:
                    rsh.write(f'''\
#!/bin/bash
#PBS -N DemEval{task[:2].upper()}{train_group[:4][2:]}{eval_group[:4][2:]}
#PBS -l walltime={walltime}
#PBS -l {node_type}
#PBS -l mem={mem},vmem={vmem}
#PBS -m be
#PBS -e {error_path}
#PBS -o {output_path}

source activate /hpf/projects/lsung/envs/anaconda/mimic4ds
cd /hpf/projects/lsung/projects/public/mimic4ds_public/mimic4ds/Experiments/baseline/scripts

python bootstrap_evaluate_by_dem.py \
--analysis_id="{task}" \
--train_group="{train_group}" \
--eval_group="{eval_group}" \
--demographic_category="{group}"
--eval='true' \
        ''')
                # run bash script
                if job_id is not None:
                    comm = f"-W depend=afterok:{job_id}"
                else:
                    comm = ''

                if use_gpu:
                    os.system(f"qsub -q gpu {comm} {script_tag}{c}.sh")
                else:
                    os.system(f'qsub {comm} {script_tag}{c}.sh')
                # remove bash script
                os.remove(f'{script_tag}{c}.sh')
    