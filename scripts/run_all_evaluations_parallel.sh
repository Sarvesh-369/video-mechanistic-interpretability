#!/bin/bash

echo "========================================================="
echo "Starting Parallel VLM Evaluations (Exp 3 - Exp 8) on 2 GPUs"
echo "========================================================="

# 1. GPU 0 Cohort (Exp 3, 4, 5) - targeting Port 8000
run_gpu0() {
    echo "[GPU 0 Cohort] Starting Exp 3 (Count x Span Sweep)..."
    python run_experiments.py exp3 --vllm-url http://localhost:8000/v1 "$@"
    
    echo "[GPU 0 Cohort] Starting Exp 4 (Temporal Position Control)..."
    python run_experiments.py exp4 --vllm-url http://localhost:8000/v1 "$@"
    
    echo "[GPU 0 Cohort] Starting Exp 5 (Matched Oracle Control)..."
    python run_experiments.py exp5 --vllm-url http://localhost:8000/v1 "$@"
}

# 2. GPU 1 Cohort (Exp 6, 7, 8) - targeting Port 8001
run_gpu1() {
    echo "[GPU 1 Cohort] Starting Exp 6 (Symbolic Evidence Control)..."
    python run_experiments.py exp6 --vllm-url http://localhost:8001/v1 "$@"
    
    echo "[GPU 1 Cohort] Starting Exp 7 (Sequence Reconstruction Task)..."
    python run_experiments.py exp7 --vllm-url http://localhost:8001/v1 "$@"
    
    echo "[GPU 1 Cohort] Starting Exp 8 (Capacity Boundary Estimation)..."
    python run_experiments.py exp8 --vllm-url http://localhost:8001/v1 "$@"
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
