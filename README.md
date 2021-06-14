# mimic4ds_public
- Scripts to reproduce "Evaluation of Domain Generalization and Adaptation on Improving Model Robustness to Temporal Dataset Shift in Clinical Medicine"

### Requirements to run:
1. A copy of the MIMIC-IV (v1.0) database on Google BigQuery (GBQ)
2. a service account key file to query GBQ tables
3. Install utils_prediction module:
    - We recommend seting up a conda environment: `conda create -p path-to-environment python=3.8`
    - cd into utils_prediction
    - do `python -m pip install -e .`
4. Change paths in scripts
