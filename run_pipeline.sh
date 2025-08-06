#!/usr/bin/bash

# This shell script reproduces the results of the paper "An interpretable machine learning model for seasonal precipitation forecasting" 
# by E. Pinheiro and T. B. M. J. Ouarda
# -n argument sets the number of samples to be used in the analysis. The original paper uses 1000 samples, but running with 100 samples gives similar results

python -W ignore feature_pre_selection.py -n 100
python -W ignore sample_model_selection.py -n 100

# Selection of the best model configuration is done by analyzing the results of the model selection procedure.
# After selecting the best model, run the following command passing the argument -c with the number of the selected configuration:
# Example, if the best model is the configuration 3, run:
# python -W ignore sample_test.py -n 100 -c 3
