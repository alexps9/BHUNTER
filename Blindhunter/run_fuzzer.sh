#!/bin/bash
# Run the BHUNTER scenario fuzzer. Execute this script from any directory;
# it switches to the Blindhunter package root before launching Python.

cd "$(dirname "$0")"

CORPUS="data/corpus/seeds"
WORKDIR="output"
MAP="Town02"
STRATEGY="guided"
DESIRED_OCCLUSION="0.8"
CARLA_PATH="${CARLA_PATH:-../carla/CarlaUE4.sh}"
TRACKING_PATH="output/tracking"
EXP_PATH="output/exp"
BENCHMARK_PATH="output/benchmark"

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
            echo "Usage: $0 [--corpus DIR] [--workdir DIR] [--map NAME] [--strategy guided|random|no_scheduling] [--desired_occlusion FLOAT] [--carla_path PATH] [--tracking_path DIR] [--exp_path DIR] [--benchmark_path DIR]"
            exit 1
            ;;
    esac
done

mkdir -p "$CORPUS" "$WORKDIR" "$TRACKING_PATH" "$EXP_PATH" "$BENCHMARK_PATH"

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

status=$?
if [ $status -eq 0 ]; then
    echo "Fuzzer completed successfully"
else
    echo "Fuzzer failed with error code $status"
    exit $status
fi
