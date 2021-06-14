"""
Generate and submit bash scripts to evaluate trained models
"""

import os

#------------------------------------------------------------------------
# PBS vars
#------------------------------------------------------------------------
script_tag = 're'

# resources
walltime = "23:00:00"
ppn = "8"
mem = "50g"
vmem = "50g"
node_type = f"nodes=1:ppn={ppn}"
use_gpu = False

# error & output paths
error_path = "/hpf/projects/lsung/projects/public/mimic4ds_public/mimic4ds/Experiments/domain_adapt/logs/bootstrap_evaluate_ls/error"
output_path = "/hpf/projects/lsung/projects/public/mimic4ds_public/mimic4ds/Experiments/domain_adapt/logs/bootstrap_evaluate_ls/output"

if not os.path.exists(error_path): os.makedirs(error_path)
if not os.path.exists(output_path): os.makedirs(output_path)
    
# Job dependencies
# -- If not None, submitted jobs will be held until the {job_id} completes 
# -- without error.
job_id = None#'64216143:64216142:64216141:64216105:64216106:64216107:64216108:64216109:64216110'


#------------------------------------------------------------------------
# Tasks
#------------------------------------------------------------------------
tasks = ['longlos','mortality','invasivevent','sepsis']
algos = ['coral','al_layer']
Ns = [100,500,1000,1500]
evaluation_methods = ['avg','ensemble','best']
lambdas_coral = [0.01, 0.1, 1, 10, 100, 1000]
lambdas_al = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1]
                    
#------------------------------------------------------------------------
# Create and submit jobs
#------------------------------------------------------------------------
c = 0
for task in tasks:
    for n in Ns:
        for algo in algos:
            for method in evaluation_methods:
                for l in range(len(lambdas_coral)):
                    if algo == 'coral': 
                        lambd = lambdas_coral[l]
                    else:
                        lambd = lambdas_al[l]
                    c+=1
                    # create bash script
                    with open (f"{script_tag}{c}.sh", 'w') as rsh:
                            rsh.write(f'''\
#!/bin/bash
#PBS -N DAeval{task[:2].upper()}{algo[:2]}{method[:4].upper()}{n}{lambd}
#PBS -l walltime={walltime}
#PBS -l {node_type}
#PBS -l mem={mem},vmem={vmem}
#PBS -m be
#PBS -e {error_path}
#PBS -o {output_path}

source activate /hpf/projects/lsung/envs/anaconda/mimic4ds
cd /hpf/projects/lsung/projects/public/mimic4ds_public/mimic4ds/Experiments/domain_adapt/scripts

python bootstrap_evaluate.py \
--analysis_id="{task}" \
--train_method="{algo}" \
--evaluation_method='{method}' \
--n_ood={n} \
--lambd={lambd} 
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
