# CLSVAE for Systematic Error Repair

A semi-supervised VAE model for outlier detection and data repair of systematic errors in dirty datasets.
Here we introduce the pytorch implementation of CLSVAE (Clean Subspace Variational Autoencoder).

This repo is the public release code for the pre-print **"Repairing Systematic Outliers by Learning Clean Subspaces in VAEs"**.\
Link to the arXiv paper here: https://arxiv.org/abs/2207.08050 . 

See paper for details on models, hyperparameters and datasets.

**Please consider citing us if you use any part of our code.**

## Instalation
- Requires Python 3.8. or higher

- Pytorch framework (v1.8.1) was used

- Used python packages can be found in <code> ./src/requirements.txt </code>
    - e.g. you can install via  <code> pip install -r  requirements.txt</code> inside your "venv" or "conda" environment

- Please install models package using inside your virtual environment (dev mode): <code>pip install -e ./src/</code>
    - this package (name is <code>repair_syserr_models</code>) contains the code for the VAE models and associated \
      utility functions
     
    - five models provided (used in paper): VAE, CVAE, VAE_GMM, CCVAE, and **CLSVAE**.

## Usage
- Jupyter notebooks with examples for all models are found in <code> ./src/notebooks/ </code>
    - the current notebooks already have training run information in the cells, visualization of metrics and repairs, but can be re-run by the user. 

- Simple bash commands to run models can be found in <code> ./src/repair_syserr_models/run_train_model.sh </code>

- An example exists (notebook, or in script) for each dataset from paper, for each model from paper,
  for 35% corruption level

- Note <code> --cuda-on </code> flag for GPU training, remove for CPU only training

## Inputs
Input data (dirty and clean datasets) for experiments to run models for notebooks and scripts in **Usage**.\
Please see below to get data.

### Data for Examples (Jupyter Notebooks and Scripts)
- Copy folder contents from <code>data</code> in Google Drive
  (available [here](https://drive.google.com/drive/folders/1YseCgYtloWd1DVpAbet-YfR0cb8x0vh1?usp=sharing))
  to your local repo folder in <code>./data/</code>

- Three datasets (Fashion MNIST, Frey Faces, Synthetic Shapes) with 35% corruption level for each, both \
  ground-truth and corrupt data version therein, and several sizes of trusted set.

## Outputs
- The output results of the training run (e.g. metrics, performance and model parameters) are then found \
  in folder <code>./outputs/experiments_test/</code>

- The current folder already includes outputs from the existing example training runs.

## License

 MIT
