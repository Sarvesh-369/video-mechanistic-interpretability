#!/bin/bash

echo "========================================================="
echo "Starting Parallel VLM Evaluations (Exp 3 - Exp 8) on 2 GPUs"
echo "========================================================="

# 1. GPU 0 Cohort (Exp 3, 4, 5)
run_gpu0() {
    echo "[GPU 0] Starting Exp 3 (Count x Span Sweep)..."
    CUDA_VISIBLE_DEVICES=0 python run_experiments.py exp3 --device cuda "$@"
    
    echo "[GPU 0] Starting Exp 4 (Temporal Position Control)..."
    CUDA_VISIBLE_DEVICES=0 python run_experiments.py exp4 --device cuda "$@"
    
    echo "[GPU 0] Starting Exp 5 (Matched Oracle Control)..."
    CUDA_VISIBLE_DEVICES=0 python run_experiments.py exp5 --device cuda "$@"
}

# 2. GPU 1 Cohort (Exp 6, 7, 8)
run_gpu1() {
    echo "[GPU 1] Starting Exp 6 (Symbolic Evidence Control)..."
    CUDA_VISIBLE_DEVICES=1 python run_experiments.py exp6 --device cuda "$@"
    
    echo "[GPU 1] Starting Exp 7 (Sequence Reconstruction Task)..."
    CUDA_VISIBLE_DEVICES=1 python run_experiments.py exp7 --device cuda "$@"
    
    echo "[GPU 1] Starting Exp 8 (Capacity Boundary Estimation)..."
    CUDA_VISIBLE_DEVICES=1 python run_experiments.py exp8 --device cuda "$@"
}

# Launch both runs in the background
run_gpu0 "$@" &
PID_GPU0=$!

run_gpu1 "$@" &
PID_GPU1=$!

echo "--> Evaluations running in background. PIDs: GPU 0=$PID_GPU0, GPU 1=$PID_GPU1"
echo "Waiting for runs to complete..."

# Wait and capture exit statuses
wait $PID_GPU0
STATUS_GPU0=$?

wait $PID_GPU1
STATUS_GPU1=$?

if [ $STATUS_GPU0 -ne 0 ] || [ $STATUS_GPU1 -ne 0 ]; then
    echo "========================================================="
    echo "❌ Error: One of the GPU evaluation runs failed!"
    echo "GPU 0 Exit Status: $STATUS_GPU0"
    echo "GPU 1 Exit Status: $STATUS_GPU1"
    echo "========================================================="
    exit 1
fi

echo "========================================================="
echo "✓ All evaluations on GPU 0 and GPU 1 completed successfully!"
echo "========================================================="

echo ""
echo "========================================================="
echo "Compiling all plotting results..."
echo "========================================================="
python src/plot_new_results.py

echo "========================================================="
echo "✓ Done! All plots are saved in results/new_plots/"
echo "========================================================="
