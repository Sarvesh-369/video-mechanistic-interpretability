#!/bin/bash
# Script to generate custom video datasets for Exp 3, 4, 5, and 8.

OUT_DIR="videos/custom_exp"
mkdir -p "$OUT_DIR"

echo "========================================================="
echo "Generating custom video datasets for Interpretability Suite"
echo "========================================================="

# 1. Exp 3: Count (2,4,6,8) x Span (4,8,16,24)
echo "--> Generating Exp 3: Count x Span (16 combinations, 5 seeds each)..."
for count in 2 4 6 8; do
    for span in 4 8 16 24; do
        for seed in {0..4}; do
            python src/generate_custom_datasets.py \
                --experiment 3 \
                --count "$count" \
                --span "$span" \
                --seed "$seed" \
                --out-dir "$OUT_DIR"
        done
    done
done

# 2. Exp 4: Temporal Position Control
echo "--> Generating Exp 4: Temporal Position (3 positions, 5 seeds each)..."
for position in early mid late; do
    for seed in {0..4}; do
        python src/generate_custom_datasets.py \
            --experiment 4 \
            --count 4 \
            --span 8.0 \
            --position "$position" \
            --seed "$seed" \
            --out-dir "$OUT_DIR"
    done
done

# 3. Exp 5: Timing Regularity Control
echo "--> Generating Exp 5: Regularity (periodic, jittered, irregular; 5 seeds each)..."
for regularity in periodic jittered irregular; do
    for seed in {0..4}; do
        python src/generate_custom_datasets.py \
            --experiment 5 \
            --count 4 \
            --span 8.0 \
            --regularity "$regularity" \
            --seed "$seed" \
            --out-dir "$OUT_DIR"
    done
done

# 4. Exp 8: High-powered Replication cohort (N=3,4,5,6,7,8; 40 seeds each)
echo "--> Generating Exp 8: Replication Cohort (6 counts, 40 seeds each)..."
for count in 3 4 5 6 7 8; do
    for seed in {0..39}; do
        python src/generate_custom_datasets.py \
            --experiment 8 \
            --count "$count" \
            --seed "$seed" \
            --out-dir "$OUT_DIR"
    done
done

echo "========================================================="
echo "✓ Custom video generation completed successfully!"
echo "========================================================="
