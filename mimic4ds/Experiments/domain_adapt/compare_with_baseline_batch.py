"""
Generate and submit bash scripts to compared models trained using domain daptation methods
w/ models (08 - 10 & 17 - 19) trained in the baseline experiment
"""

import os

#------------------------------------------------------------------------
# PBS vars
#------------------------------------------------------------------------
script_tag = 'r'

# resources
walltime = "23:00:00"
ppn = "8"
mem = "50g"
vmem = "50g"
node_type = f"nodes=1:ppn={ppn}"
use_gpu = False

# error & output paths
error_path = "/hpf/projects/lsung/projects/public/mimic4ds_public/mimic4ds/Experiments/domain_adapt/logs/bootstrap_compare_with_baseline/error"
output_path = "/hpf/projects/lsung/projects/public/mimic4ds_public/mimic4ds/Experiments/domain_adapt/logs/bootstrap_compare_with_baseline/output"

if not os.path.exists(error_path): os.makedirs(error_path)
if not os.path.exists(output_path): os.makedirs(output_path)
    
# Job dependencies
# -- If not None, submitted jobs will be held until the {job_id} completes 
# -- without error.
job_id = None#'63620060:63620061:63620062:63620063'


#------------------------------------------------------------------------
# Tasks
#------------------------------------------------------------------------
tasks = ['longlos','mortality','invasivevent','sepsis']
train_methods = ['coral','al_layer']
n_oods = [100,500,1000,1500]
comparison_models = ['base','oracle']
                    
#------------------------------------------------------------------------
# Create and submit jobs
#------------------------------------------------------------------------
c = 0
for task in tasks:
    for model in comparison_models:
        for n_ood in n_oods:
            for method in train_methods:
                c+=1
                # create bash script
                with open (f"{script_tag}{c}.sh", 'w') as rsh:
                        rsh.write(f'''\
#!/bin/bash
#PBS -N DAcomp{task[:2].upper()}{model}{n_ood}{method[:3].upper()}
#PBS -l walltime={walltime}
#PBS -l {node_type}
#PBS -l mem={mem},vmem={vmem}
#PBS -m be
#PBS -e {error_path}
#PBS -o {output_path}

source activate /hpf/projects/lsung/envs/anaconda/mimic4ds
cd /hpf/projects/lsung/projects/public/mimic4ds_public/mimic4ds/Experiments/domain_adapt/scripts

python bootstrap_compare_with_baselineExp.py \
--analysis_id="{task}" \
--train_method='{method}' \
--n_ood={n_ood} \
--comparison_model='{model}' \
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
