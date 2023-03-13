# CLSVAE for Systematic Error Repair

A semi-supervised VAE model for outlier detection and repair of systematic errors in dirty datasets.

This repo is the public code for the arxiv pre-print "CLSVAE: Clean Subspace VAE for Systematic Error Repair".
Check it here https://arxiv.org/abs/2207.08050 .
**Please consider citing us if you use any part of our code.**

## Instalation
- Requires Python 3.8. or higher

- Used python packages can be found in <code> ./src/requirements.txt </code>
    - e.g. install via  <code> pip install -r  requirements.txt</code> inside your "venv" or "conda" environment

- Please install models package using inside your virtual environment (dev mode): <code>pip install -e ./src/</code>
    - this contains the code for the VAE models and associated utility functions

- Example jupyter notebooks with examples for all models are found in <code> ./src/notebooks/ </code>

- Simple bash commands to run models can be found in <code> ./src/run_train_model.sh </code>

- Note <code> --cuda-on </code> flag for GPU training, remove for CPU

CONTINUE AND FINISH README.md !!
