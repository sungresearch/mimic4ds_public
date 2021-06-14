#!/bin/bash
#PBS -N split_and_preprocess
#PBS -l walltime=96:00:00
#PBS -l nodes=1:ppn=16
#PBS -l mem=100g,vmem=100g
#PBS -m e
#PBS -e /hpf/projects/lsung/projects/mimic4ds/Experiments/baseline/logs/load_and_preprocess/error
#PBS -o /hpf/projects/lsung/projects/mimic4ds/Experiments/baseline/logs/load_and_preprocess/output

source activate /hpf/projects/lsung/envs/anaconda/mimic4ds
cd /hpf/projects/lsung/projects/public/mimic4ds_public/mimic4ds/Experiments/baseline/scripts

## Create and loop through tasks and groups
declare -a tasks=("invasivevent" "sepsis" "longlos" "mortality")
declare -a groups=("all" "2008 - 2010" "2011 - 2013" "2014 - 2016" "2017 - 2019")
for analysis_id in "${tasks[@]}"
    do for group in "${groups[@]}"
        do python split_and_fit_preprocessors.py \
            --analysis_id="$analysis_id" \
            --group="$group" \
            --debug="false"
    done
done