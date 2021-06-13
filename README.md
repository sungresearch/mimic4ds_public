# mimic4ds_public

## TODO: 
- upload experiment scripts
- update readme
- clean up

### Requirements:
1. A copy of the MIMIC-IV (v1.0) database on Google BigQuery (GBQ)
2. a service account key file to query GBQ tables

### Setup:
1. Clone respository - `git clone https://github.com/sungresearch/mimic4ds_public.git`
2. Create conda environment - `conda create -p path-to-environment python=3.8`
2. Install utils_prediction
	a. cd into utils_prediction
	b. do `pip install -e .`
3. Edit paths in scripts and service account key file
