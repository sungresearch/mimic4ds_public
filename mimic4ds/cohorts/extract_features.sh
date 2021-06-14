## Can't submit job via qsub - compute nodes do not allow external connections (required for gbq)
## to run: ./extract_features.sh 
## extracts features from gbq and save to mimic4ds/data 

cd /hpf/projects/lsung/projects/public/mimic4ds_public/mimic4ds

config_path="/hpf/projects/lsung/projects/public/mimic4ds_public/mimic4ds/cohorts/featurizer_configs"
save_path="/hpf/projects/lsung/projects/public/mimic4ds_public/mimic4ds/data"

## extract features
python extract_features.py \
    --save_path="$save_path" \
    --config_path="$config_path/mortality.yml"

python extract_features.py \
    --save_path="$save_path" \
    --config_path="$config_path/longlos.yml"

python extract_features.py \
    --save_path="$save_path" \
    --config_path="$config_path/sepsis.yml"

python extract_features.py \
    --save_path="$save_path" \
    --config_path="$config_path/invasivevent.yml"
    
