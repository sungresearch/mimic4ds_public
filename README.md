# mimic4ds_public
- Scripts to reproduce "Evaluation of Domain Generalization and Adaptation on Improving Model Robustness to Temporal Dataset Shift in Clinical Medicine"

<<<<<<< HEAD
### Requirements to run:
1. A copy of the MIMIC-IV (v1.0) database on Google BigQuery (GBQ)
2. a service account key file to query GBQ tables
3. Install utils_prediction module:
    - We recommend seting up a conda environment: `conda create -p path-to-environment python=3.8`
    - cd into utils_prediction
    - do `python -m pip install -e .`
4. Change paths in scripts
=======
## TODO: 
- upload experiment scripts
- update readme
- clean up

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
>>>>>>> f0b6622a14dcbadcba7cd69389b0ccb7e74e061d
