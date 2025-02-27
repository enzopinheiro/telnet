#!/usr/bin/bash

# 1) This shell script reproduces the results of the paper "TelNet: An interpretable machine learning model for seasonal precipitation forecasting" 
# by E. Pinheiro and T. B. M. J. Ouarda
# 2) It requires about 16GB of VRAM and takes about a 4 days to run the full sampling procedure on a single A100 GPU
# 3) Using different versions of the libraries may lead to different results due to stochastic nature of the algorithms, 
# thus it is recommended to use the container provided in the root directory. Further instructions on how to mount and use the container 
# are provided in docker directory
# 4) The dataset used in the paper is not provided in the root directory, but can be downloaded from the following link: 
# https://drive.google.com/file/d/1ZZvyF-nRT_k7aqFCDpOBCMt0_UenRT-W/view?usp=sharing
# 5) Once the dataset is downloaded, an environmental variable "DATADIR" should be set to the path of the dataset

python -W ignore sample_model_selection.py -n 1000  # Original paper runs with 1000 samples, but running with 100 samples gives similar results
python -W ignore sample_test.py -n 1000  # Original paper runs with 1000 samples, but running with 100 samples gives similar results
