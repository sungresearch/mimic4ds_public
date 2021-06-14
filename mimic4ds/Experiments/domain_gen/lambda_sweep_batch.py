"""
Generate and submit bash scripts to train model using specified training
algorithm and lambda group regularization hyperparameter
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
node_type = f"nodes=1:ppn={ppn}:gpus=1"
use_gpu = True

# error & output paths
error_path = "/hpf/projects/lsung/projects/public/mimic4ds_public/mimic4ds/Experiments/domain_gen/logs/lambda_sweep/error"
output_path = "/hpf/projects/lsung/projects/public/mimic4ds_public/mimic4ds/Experiments/domain_gen/logs/lambda_sweep/output"

if not os.path.exists(error_path): os.makedirs(error_path)
if not os.path.exists(output_path): os.makedirs(output_path)
    
# Job dependencies
# -- If not None, submitted jobs will be held until the {job_id} completes 
# -- without error.
job_id = None#"63714366"


#------------------------------------------------------------------------
# Tasks
#------------------------------------------------------------------------
tasks = ['longlos', 'mortality','sepsis','invasivevent'] 
methods = ['dro','irm','coral','al_layer']
lambdas_coral = [0.01, 0.1, 1, 10, 100, 1000]
lambdas_irm = [0.1, 1, 10, 100, 1000, 10000]
lambdas_dro = [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1]
lambdas_al = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1]

                    
#------------------------------------------------------------------------
# Create and submit jobs
#------------------------------------------------------------------------
c = 0
for task in tasks:
    for method in methods:
        for l in range(len(lambdas_coral)):
            c+=1
            if method == 'coral': 
                lambd = lambdas_coral[l]
            elif method == 'dro':
                lambd = lambdas_dro[l]
            elif method == 'irm':
                lambd = lambdas_irm[l]
            else:
                lambd = lambdas_al[l]
            # create bash script
            with open (f"{script_tag}{c}.sh", 'w') as rsh:
                rsh.write(f'''\
#!/bin/bash
#PBS -N ls{task[:2].upper()}{method}
#PBS -l walltime={walltime}
#PBS -l {node_type}
#PBS -l mem={mem},vmem={vmem}
#PBS -m be
#PBS -e {error_path}
#PBS -o {output_path}

source activate /hpf/projects/lsung/envs/anaconda/mimic4ds
cd /hpf/projects/lsung/projects/public/mimic4ds_public/mimic4ds/Experiments/domain_gen/scripts

python train.py \
--analysis_id="{task}" \
--train_method="{method}" \
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
    