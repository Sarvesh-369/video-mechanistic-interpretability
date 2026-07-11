import os
import argparse
import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def bootstrap_confidence_interval(data, num_bootstraps=1000, ci=0.95):
    """
    Computes bootstrapped confidence interval for binary accuracy data.
    """
    if len(data) == 0:
        return 0.0, 0.0
    bootstraps = []
    for _ in range(num_bootstraps):
        sample = np.random.choice(data, size=len(data), replace=True)
        bootstraps.append(np.mean(sample))
    lower = np.percentile(bootstraps, (1 - ci) / 2 * 100)
    upper = np.percentile(bootstraps, (1 + ci) / 2 * 100)
    return lower, upper

def plot_exp3_heatmap(results_json, output_dir):
    """
    Generates 2D heatmap for Count x Span.
    """
    if not os.path.exists(results_json):
        print(f"Skipping Exp 3 plot: {results_json} not found.")
        return
        
    with open(results_json, "r") as f:
        data = json.load(f)
        
    # Group by count and span
    grid = {}
    for item in data:
        N = item["count"]
        S = item["span"]
        corr = 1.0 if item["correct"] else 0.0
        grid.setdefault(N, {}).setdefault(S, []).append(corr)
        
    counts = sorted(list(grid.keys()))
    spans = sorted(list({S for N in grid for S in grid[N]}))
    
    if not counts or not spans:
        return
        
    heatmap_data = np.zeros((len(counts), len(spans)))
    for i, N in enumerate(counts):
        for j, S in enumerate(spans):
            acc_list = grid[N].get(S, [])
            heatmap_data[i, j] = np.mean(acc_list) if acc_list else 0.0
            
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(heatmap_data, cmap="Blues", vmin=0, vmax=1)
    
    ax.set_xticks(range(len(spans)))
    ax.set_xticklabels([f"{S}s" for S in spans], fontweight="bold")
    ax.set_yticks(range(len(counts)))
    ax.set_yticklabels([f"N={N}" for N in counts], fontweight="bold")
    
    ax.set_xlabel("Active Temporal Span", fontsize=11, fontweight="bold")
    ax.set_ylabel("Event Count (N)", fontsize=11, fontweight="bold")
    ax.set_title("VLM Counting Accuracy (Count x Span Grid)", fontsize=13, fontweight="bold", pad=15)
    
    # Annotate cells
    for i in range(len(counts)):
        for j in range(len(spans)):
            val = heatmap_data[i, j]
            ax.text(j, i, f"{val*100:.1f}%", ha="center", va="center", 
                    color="white" if val > 0.5 else "black", fontweight="bold")
            
    fig.colorbar(im, label="Accuracy")
    plt.tight_layout()
    out_path = Path(output_dir) / "exp3_count_span_heatmap.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"✓ Saved Exp 3 heatmap to {out_path}")

def plot_exp4_position(results_json, output_dir):
    """
    Generates bar chart comparing temporal positions.
    """
    if not os.path.exists(results_json):
        print(f"Skipping Exp 4 plot: {results_json} not found.")
        return
        
    with open(results_json, "r") as f:
        data = json.load(f)
        
    pos_grid = {}
    for item in data:
        pos = item["position"]
        corr = 1.0 if item["correct"] else 0.0
        pos_grid.setdefault(pos, []).append(corr)
        
    positions = ["early", "mid", "late"]
    accuracies = [np.mean(pos_grid.get(p, [0.0])) for p in positions]
    
    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(positions, accuracies, color=["#3498db", "#2ecc71", "#e74c3c"], width=0.5, edgecolor="grey")
    
    ax.set_ylabel("Average Accuracy", fontsize=11, fontweight="bold")
    ax.set_xlabel("Temporal Position of Active Span", fontsize=11, fontweight="bold")
    ax.set_title("Effect of Event Temporal Position on Accuracy", fontsize=13, fontweight="bold", pad=15)
    ax.set_ylim(0, 1.1)
    
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f"{yval*100:.1f}%", ha='center', va='bottom', fontweight="bold")
        
    plt.tight_layout()
    out_path = Path(output_dir) / "exp4_position_control.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"✓ Saved Exp 4 position chart to {out_path}")

def plot_controls_comparison(exp3_json, exp5_json, exp6_json, output_dir):
    """
    Generates comparative bar chart between Video, Oracle, and Symbolic conditions.
    """
    if not os.path.exists(exp3_json) or not os.path.exists(exp5_json) or not os.path.exists(exp6_json):
        print("Skipping controls comparison plot: missing one or more of Exp 3, 5, 6 results.")
        return
        
    with open(exp3_json, "r") as f:
        data_v = json.load(f)
    with open(exp5_json, "r") as f:
        data_o = json.load(f)
    with open(exp6_json, "r") as f:
        data_s = json.load(f)
        
    def group_by_count(dataset):
        grid = {}
        for item in dataset:
            grid.setdefault(item["count"], []).append(1.0 if item["correct"] else 0.0)
        return {c: np.mean(vals) for c, vals in grid.items()}
        
    acc_v = group_by_count(data_v)
    acc_o = group_by_count(data_o)
    acc_s = group_by_count(data_s)
    
    counts = sorted(list(set(acc_v.keys()) | set(acc_o.keys()) | set(acc_s.keys())))
    if not counts:
        return
        
    val_v = [acc_v.get(c, 0.0) for c in counts]
    val_o = [acc_o.get(c, 0.0) for c in counts]
    val_s = [acc_s.get(c, 0.0) for c in counts]
    
    x = np.arange(len(counts))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.bar(x - width, val_v, width, label="Raw Video (Standard)", color="#e74c3c", edgecolor="grey")
    ax.bar(x, val_o, width, label="Matched-Length Oracle", color="#f1c40f", edgecolor="grey")
    ax.bar(x + width, val_s, width, label="Symbolic Evidence (Text-Only)", color="#2ecc71", edgecolor="grey")
    
    ax.set_ylabel("Accuracy", fontsize=11, fontweight="bold")
    ax.set_xlabel("Event Count (N)", fontsize=11, fontweight="bold")
    ax.set_title("VLM Counting: Video vs. Oracle vs. Symbolic Cues", fontsize=13, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels([f"N={c}" for c in counts], fontweight="bold")
    ax.set_ylim(0, 1.1)
    ax.legend()
    
    plt.tight_layout()
    out_path = Path(output_dir) / "exp5_6_controls_comparison.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"✓ Saved controls comparison chart to {out_path}")

def plot_exp7_reconstruction(results_json, output_dir):
    """
    Generates accuracy comparison for reconstruction/order verification tasks.
    """
    if not os.path.exists(results_json):
        print(f"Skipping Exp 7 plot: {results_json} not found.")
        return
        
    with open(results_json, "r") as f:
        data = json.load(f)
        
    order_correct_list = []
    seq_match_list = []
    
    for item in data:
        order_correct_list.append(1.0 if item["order_correct"] else 0.0)
        # Sequence matches count if generated bounces equals true count
        # In Exp 7 results we saved count as metadata in evaluate script
        # Wait, if we didn't save count in evaluation script: let's verify!
        # Ah, we did: results.append({"count": count, "seq_match_count": occurrences...})
        gt_count = item.get("count", 4)
        pred_occurrences = item["seq_match_count"]
        seq_match_list.append(1.0 if pred_occurrences == gt_count else 0.0)
        
    tasks = ["Order Verification (Yes/No)", "Sequence Count Reconstruction"]
    accuracies = [np.mean(order_correct_list) if order_correct_list else 0.0,
                  np.mean(seq_match_list) if seq_match_list else 0.0]
                  
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(tasks, accuracies, color=["#9b59b6", "#34495e"], width=0.4, edgecolor="grey")
    
    ax.set_ylabel("Accuracy / Match Rate", fontsize=11, fontweight="bold")
    ax.set_title("Non-Count Temporal Bookkeeping & Order Verification", fontsize=13, fontweight="bold", pad=15)
    ax.set_ylim(0, 1.1)
    
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f"{yval*100:.1f}%", ha='center', va='bottom', fontweight="bold")
        
    plt.tight_layout()
    out_path = Path(output_dir) / "exp7_bookkeeping_accuracy.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"✓ Saved Exp 7 bookkeeping chart to {out_path}")

def plot_exp8_boundary(results_json, output_dir):
    """
    Plots accuracy line curve across counts with 95% bootstrapped confidence intervals.
    Estimates the exact performance cutoff boundary.
    """
    if not os.path.exists(results_json):
        print(f"Skipping Exp 8 plot: {results_json} not found.")
        return
        
    with open(results_json, "r") as f:
        data = json.load(f)
        
    grid = {}
    for item in data:
        N = item["count"]
        corr = 1.0 if item["correct"] else 0.0
        grid.setdefault(N, []).append(corr)
        
    counts = sorted(list(grid.keys()))
    if not counts:
        return
        
    accuracies = []
    low_bounds = []
    high_bounds = []
    
    for N in counts:
        corr_list = grid[N]
        acc = np.mean(corr_list)
        low, high = bootstrap_confidence_interval(corr_list)
        accuracies.append(acc)
        low_bounds.append(low)
        high_bounds.append(high)
        
    # Estimate boundary cutoff: largest N where lower 95% CI bound is >= 0.8
    boundary_cutoff = None
    for i, N in enumerate(counts):
        if low_bounds[i] >= 0.8:
            boundary_cutoff = N
            
    fig, ax = plt.subplots(figsize=(8, 5))
    
    ax.plot(counts, accuracies, color="#1abc9c", marker="o", linewidth=2.5, label="VLM Accuracy")
    ax.fill_between(counts, low_bounds, high_bounds, color="#1abc9c", alpha=0.2, label="95% Bootstrapped CI")
    
    # Draw standard threshold line (e.g. 0.8)
    ax.axhline(y=0.8, color="#e74c3c", linestyle=":", alpha=0.7, label="Acceptable Threshold (0.8)")
    
    if boundary_cutoff is not None:
        ax.axvline(x=boundary_cutoff, color="#2c3e50", linestyle="--", alpha=0.8, 
                   label=f"Estimated Capacity Boundary (N={boundary_cutoff})")
                   
    ax.set_xlabel("Event Count (N)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Accuracy", fontsize=11, fontweight="bold")
    ax.set_title("Replication Capacity Curve & Estimated Counting Boundary", fontsize=13, fontweight="bold", pad=15)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower left")
    
    plt.tight_layout()
    out_path = Path(output_dir) / "exp8_boundary_estimation.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"✓ Saved Exp 8 boundary estimation plot to {out_path}")

def main():
    parser = argparse.ArgumentParser(description="Compile and plot all new experiment results")
    parser.add_argument("--results-dir", type=str, default="results/new_results", help="Directory of evaluation JSON results")
    parser.add_argument("--output-dir", type=str, default="results/new_plots", help="Directory to save generated charts")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    results_dir = Path(args.results_dir)
    
    # 1. Plot Exp 3 Heatmap
    plot_exp3_heatmap(results_dir / "exp3_results.json", args.output_dir)
    
    # 2. Plot Exp 4 Position Controls
    plot_exp4_position(results_dir / "exp4_results.json", args.output_dir)
    
    # 3. Plot Exp 5 & 6 Controls Comparison
    plot_controls_comparison(
        results_dir / "exp3_results.json",
        results_dir / "exp5_results.json",
        results_dir / "exp6_results.json",
        args.output_dir
    )
    
    # 4. Plot Exp 7 Sequence Reconstruction Tasks
    plot_exp7_reconstruction(results_dir / "exp7_results.json", args.output_dir)
    
    # 5. Plot Exp 8 Capacity Boundary
    plot_exp8_boundary(results_dir / "exp8_results.json", args.output_dir)
    
    print(f"\n✓ All compilation plots compiled and saved to {args.output_dir}/")

if __name__ == "__main__":
    main()
