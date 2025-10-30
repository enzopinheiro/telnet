# Building the image (in the docker directory):
> sudo docker build -t telnet .

# Run the data download and preprocessing process (in the root directory):
> sudo docker run --gpus all -it --rm -v $(pwd):/code -v $CDSAPI_RC:/data/.cdsapirc -v $TELNET_DATADIR:/data --workdir /code -e PYTHONPATH=/code -e CDSAPI_RC=/data/.cdsapirc -e TELNET_DATADIR=/data telnet:latest ./data_downloader.sh

# Run the model evaluation process (in the root directory):
> sudo docker run --gpus all -it --rm -v $(pwd):/code -v $TELNET_DATADIR:/data --workdir /code -e PYTHONPATH=/code -e TELNET_DATADIR=/data telnet:latest ./model_selection.sh number_of_samples number_of_gpus

# Run the model testing process (in the root directory)
> sudo docker run --gpus all -it --rm -v $(pwd):/code -v $TELNET_DATADIR:/data --workdir /code -e PYTHONPATH=/code -e TELNET_DATADIR=/data telnet:latest python -W ignore model_testing.py -n number_of_samples -c selected_configuration

# Generate a forecast (in the root directory):
> sudo docker run --gpus all -it --rm -v $(pwd):/code -v $CDSAPI_RC:/data/.cdsapirc -v $TELNET_DATADIR:/data --workdir /code -e PYTHONPATH=/code -e CDSAPI_RC=/data/.cdsapirc -e TELNET_DATADIR=/data telnet:latest ./generate_forecast.sh initalization_date selected_configuration
