#!/bin/bash

# Default values
CORPUS="../data/corpus"
WORKDIR="../output"
MAP="Town01"
STRATEGY="guided"
DESIRED_OCCLUSION=0.8
CARLA_PATH="../carla/CarlaUE4.sh"
TRACKING_PATH="../data/tracking"
EXP_PATH="../data/exp"
BENCHMARK_PATH="../data/benchmark"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --corpus)
            CORPUS="$2"
            shift 2
            ;;
        --workdir)
            WORKDIR="$2"
            shift 2
            ;;
        --map)
            MAP="$2"
            shift 2
            ;;
        --strategy)
            STRATEGY="$2"
            shift 2
            ;;
        --desired_occlusion)
            DESIRED_OCCLUSION="$2"
            shift 2
            ;;
        --carla_path)
            CARLA_PATH="$2"
            shift 2
            ;;
        --tracking_path)
            TRACKING_PATH="$2"
            shift 2
            ;;
        --exp_path)
            EXP_PATH="$2"
            shift 2
            ;;
        --benchmark_path)
            BENCHMARK_PATH="$2"
            shift 2
            ;;
        *)
            echo "Unknown parameter: $1"
            exit 1
            ;;
    esac
done

# Create necessary directories if they don't exist
mkdir -p "$CORPUS"
mkdir -p "$WORKDIR"
mkdir -p "$TRACKING_PATH"
mkdir -p "$EXP_PATH"
mkdir -p "$BENCHMARK_PATH"

# Print configuration
echo "Starting fuzzer with the following configuration:"
echo "Corpus directory: $CORPUS"
echo "Working directory: $WORKDIR"
echo "CARLA map: $MAP"
echo "Fuzzing strategy: $STRATEGY"
echo "Desired occlusion: $DESIRED_OCCLUSION"
echo "CARLA path: $CARLA_PATH"
echo "Tracking path: $TRACKING_PATH"
echo "Experiment path: $EXP_PATH"
echo "Benchmark path: $BENCHMARK_PATH"

# Run the fuzzer
python exp_fuzzer.py \
    --corpus "$CORPUS" \
    --workdir "$WORKDIR" \
    --map "$MAP" \
    --strategy "$STRATEGY" \
    --desired_occlusion "$DESIRED_OCCLUSION" \
    --carla_path "$CARLA_PATH" \
    --tracking_path "$TRACKING_PATH" \
    --exp_path "$EXP_PATH" \
    --benchmark_path "$BENCHMARK_PATH"

# Check if the fuzzer ran successfully
if [ $? -eq 0 ]; then
    echo "Fuzzer completed successfully"
else
    echo "Fuzzer failed with error code $?"
    exit 1
fi 
