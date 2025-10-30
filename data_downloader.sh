#!/bin/bash

# Script to download and preprocess historical data for TelNet evaluation and testing.
# Requires Climate Data Store API access.
# In order to have access to Climate Data Store API, please follow this guideline https://cds.climate.copernicus.eu/how-to-api
# CDSAPI_RC and TELNET_DATADIR environment variables must be set.
# CDSAPI_RC: path to .cdsapirc file
# TELNET_DATADIR: path to the directory where data will be stored

# Download and preprocess data for TelNet evaluation.
python -W ignore download_preprocess_data.py -idate 194001 -fdate 202412

# Compute climate indices up to 2024.
python -W ignore compute_climate_indices.py -fdate 202412
