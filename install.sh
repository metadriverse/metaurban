#!/bin/bash

# conda path
source  ${CONDA_PREFIX}/etc/profile.d/conda.sh

# create conda environment
conda create -n metaurban python=3.10 -y
conda activate metaurban

# install metaurban
pip install -e .

# install additional dependencies
pip install stable_baselines3 imitation tensorboard wandb scikit-image pyyaml gdown "pybind11[global]"

# install orca
cd metaurban/orca_algo
rm -rf build/ && bash compile.sh 
cd ../..
