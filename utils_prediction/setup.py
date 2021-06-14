from setuptools import setup, find_packages

setup(
    name='utils_prediction',
    version='0.2.1',
    author='lawrence and Shahlab',
    author_email='lawrence.guo@sickkids.ca',
    packages=find_packages(),
    install_requires=[
        "jupyterlab",
        "numpy",
        "pandas>=1.0.0",
        "pyarrow",
        "scipy",
        "scikit-learn==0.24",
        "imbalanced-learn",
        "torch==1.7.1",
        "pyyaml",
        "tqdm",
        "sparse",
        "argparse",
        "joblib",
        "google-cloud-bigquery",
        "google-cloud-bigquery-storage",
        "pandas-gbq",
        "google_oauth2_tool",
        "matplotlib",
        "seaborn"
    ]
)
