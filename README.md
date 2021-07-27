# mimic4ds_public
- The code to reproduce ["Evaluation of Domain Generalization and Adaptation on Improving Model Robustness to Temporal Dataset Shift in Clinical Medicine"](https://www.medrxiv.org/content/10.1101/2021.06.17.21259092v1)

### Requirements to run:
1. A copy of the MIMIC-IV (v1.0) database on Google BigQuery (GBQ)
2. a service account key file to query GBQ tables
3. Install utils_prediction module:
    - We recommend seting up a conda environment: `conda create -p path-to-environment python=3.8`
    - cd into utils_prediction
    - do `python -m pip install -e .`
4. Change paths in scripts
