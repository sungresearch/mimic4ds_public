"""
Generate and submit bash scripts to collect results
"""

import os

#------------------------------------------------------------------------
# PBS vars
#------------------------------------------------------------------------
script_tag = 'r'

# resources
walltime = "23:00:00"
ppn = "8"
mem = "32g"
vmem = "32g"
node_type = f"nodes=1:ppn={ppn}"

# error & output paths
error_path = "/hpf/projects/lsung/projects/public/mimic4ds_public/mimic4ds/Experiments/domain_adapt/logs/collect_results/error"
output_path = "/hpf/projects/lsung/projects/public/mimic4ds_public/mimic4ds/Experiments/domain_adapt/logs/collect_results/output"

if not os.path.exists(error_path): os.makedirs(error_path)
if not os.path.exists(output_path): os.makedirs(output_path)
    
# Job dependencies
# -- If not None, submitted jobs will be held until the {job_id} completes 
# -- without error.
job_id = None


#------------------------------------------------------------------------
# Tasks
#------------------------------------------------------------------------
ps = [0.001, 0.01, 0.05, 0.1]
                    
#------------------------------------------------------------------------
# Create and submit jobs
#------------------------------------------------------------------------
c = 0
for p in ps:
    c+=1
    # create bash script
    with open (f"{script_tag}{c}.sh", 'w') as rsh:
        rsh.write(f'''\
#!/bin/bash
#PBS -N DAcollect{p}
#PBS -l walltime={walltime}
#PBS -l {node_type}
#PBS -l mem={mem},vmem={vmem}
#PBS -m be
#PBS -e {error_path}
#PBS -o {output_path}

source activate /hpf/projects/lsung/envs/anaconda/mimic4ds
cd /hpf/projects/lsung/projects/public/mimic4ds_public/mimic4ds/Experiments/domain_adapt/scripts

python collect_results.py \
--alpha={p} \
--save_tag="{p}" \
--collect_model_evaluations_by_dem='true' \
--collect_model_comparisons_by_dem='true' \
    ''')
    # run bash script
    if job_id is not None:
        comm = f"-W depend=afterok:{job_id}"
    else:
        comm = ''

    os.system(f'qsub {comm} {script_tag}{c}.sh')
    # remove bash script
    os.remove(f'{script_tag}{c}.sh')
    
