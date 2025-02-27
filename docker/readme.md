# Building the image (in the docker directory):
> sudo docker build -t telnet .

# Running the reproduce_paper.sh (in the root directory):
> sudo docker run --gpus all -it --rm -v $(pwd):/code -v $DATADIR:/data --workdir /code -e PYTHONPATH=/code -e DATADIR=/data telnet:latest ./reproduce_paper.sh
