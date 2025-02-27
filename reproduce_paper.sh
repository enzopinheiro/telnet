#!/usr/bin/bash

# This shell script reproduces the results of the paper "An interpretable machine learning model for seasonal precipitation forecasting" 
# by E. Pinheiro and T. B. M. J. Ouarda
# -n argument sets the number of samples to be used in the analysis. The original paper uses 1000 samples, but running with 100 samples gives similar results

python -W ignore sample_model_selection.py -n 1000
python -W ignore sample_test.py -n 1000
