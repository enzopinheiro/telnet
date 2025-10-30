#!/usr/bin/bash

# This shell script executes the model selection and testing pipeline for TelNet.
# It takes two arguments: the number of samples to be used in the analysis and the number of GPUs to be used for parallel processing.
# Example usage: ./model_selection_test.sh 100 4
# -n argument sets the number of samples to be used in the analysis. The original paper uses 1000 samples, but running with 100 samples gives similar results
if [ $# -lt 1 ] || [ $# -gt 2 ]; then
    echo "Usage: $0 <num_samples> [num_gpus]"
    echo "  <num_samples>  : positive integer, number of samples to use (e.g. 100)"
    echo "  [num_gpus]     : optional positive integer, number of GPUs (default: 1)"
    exit 1
fi

if ! [[ "$1" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: <num_samples> must be a positive integer."
    echo "Usage: $0 <num_samples> [num_gpus]"
    exit 1
fi

if [ -n "$2" ] && ! [[ "$2" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: [num_gpus] must be a positive integer if provided."
    echo "Usage: $0 <num_samples> [num_gpus]"
    exit 1
fi

# Ranking of features based on Partial Mutual Information score
# Inside the script, the ranking score can be changed to Partial Correlation
python -W ignore feature_pre_selection.py -n $1

# Model configuration selection using parallel processing across multiple GPUs
num_gpus=${2:-1}
if [ "$num_gpus" -lt 1 ]; then num_gpus=1; fi
total_samples=${1:-100}
chunk=$(( total_samples / num_gpus ))

echo "Starting model selection with $total_samples samples using $num_gpus GPUs..."
for ((gpu=0; gpu<num_gpus; gpu++)); do
    i=$(( gpu * chunk ))
    if [ "$gpu" -eq $((num_gpus-1)) ]; then
        f=$total_samples
    else
        f=$(( (gpu+1) * chunk ))
    fi

    python -W ignore sample_model_selection.py -n "$total_samples" -i "$i" -f "$f" -gpu "$gpu" &

    # give GPU time to initialize before starting the next batch
    if [ "$gpu" -lt $((num_gpus-1)) ]; then sleep 60; fi
done
wait
echo "Model selection completed."
