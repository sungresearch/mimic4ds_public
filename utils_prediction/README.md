## Utility functions for predictions

#### Installation guide:
1. Clone the repository - git clone https://github.com/sungresearch/utils_prediction.git
2. do `pip install -e .` from within the directory

- if installation throws error related to PyTorch version being not found, please install PyTorch following the guide on https://pytorch.org/ with the appropriate CUDA version for your system. The CUDA version for SickKids HPC is 10.1. 

#### Modules:
- `database`: API and utilities to connect to and query databases 
- `featurizer`: Feature extraction from databases (currently supports feature extraction from MIMIC-4 on GBQ)
- `dataloader`: Dataloader class that loads extracted features, splits them into datasets, and converts them into formats that work with other modules (e.g., nn)
- `preprocessor`: Scikit-learn compatible preprocessors
- `nn`: Pytorch models for supervised learning
- `model_selection`: Utilities for model selection
- `model_evaluation`: Utilities for model evaluation
- `model_inspection`: Utilities for model inspection
- `demos`: Example notebooks
