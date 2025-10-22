#!/usr/bin/bash

# This shell script reproduces the results of the paper "An interpretable machine learning model for seasonal precipitation forecasting" 
# by E. Pinheiro and T. B. M. J. Ouarda
# -n argument sets the number of samples to be used in the analysis. The original paper uses 1000 samples, but running with 100 samples gives similar results

# Feature pre-selection
python -W ignore feature_pre_selection.py -n 100

# # Model configuration selection (serial with a single GPU)
# python -W ignore sample_model_selection.py -n 100
# Run the following if you want to run model selection in parallel on multiple GPUs
# Example for running model selection on 100 samples distributed across 4 GPUs (0, 1, 2, 3):
# Run first batch (samples 0-24) on GPU 0 in background
python -W ignore sample_model_selection.py -n 100 -i 0 -f 24 -gpu 0 &
# Wait 60 seconds to let GPU 0 initialize and create the necessary files
sleep 60
# Run the remaining three batches in parallel on GPUs 1, 2, 3
python -W ignore sample_model_selection.py -n 100 -i 25 -f 49 -gpu 1 &
python -W ignore sample_model_selection.py -n 100 -i 50 -f 74 -gpu 2 &
python -W ignore sample_model_selection.py -n 100 -i 75 -f 99 -gpu 3 &

# Selection of the best model configuration is done by analyzing the results of the model selection procedure.
# After selecting the best model, run the following command passing the argument -c with the number of the selected configuration:
# Example, if the best model is the configuration 3, run:
# python -W ignore sample_test.py -n 100 -c 3
