import os
import argparse
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from src.utils.model_helpers import (
    load_model_and_processor, 
    prepare_video_inputs, 
    find_video_files, 
    get_associated_files, 
    extract_representation_trajectory,
    format_prompt_by_mode
)

def compute_trajectory_metrics(trajectory):
    """
    Computes consecutive frame cosine similarities (both raw and mean-centered/correlation) 
    and trajectory drift.
    """
    T, hidden_dim = trajectory.shape
    
    # --- 1. Raw Cosine Similarities ---
    norms = np.linalg.norm(trajectory, axis=1, keepdims=True)
    norm_trajectory = trajectory / (norms + 1e-9)
    
    consecutive_sims = []
    for t in range(T - 1):
        sim = np.dot(norm_trajectory[t], norm_trajectory[t+1])
        consecutive_sims.append(float(sim))
        
    init_sims = []
    for t in range(T):
        sim = np.dot(norm_trajectory[t], norm_trajectory[0])
        init_sims.append(float(sim))
        
    # --- 2. Mean-Centered Cosine Similarities (Pearson Correlation) ---
    # Subtracting the average representation across time removes the static background
    # and the massive common anisotropy vector (the cone effect).
    mean_vector = np.mean(trajectory, axis=0)
    centered_trajectory = trajectory - mean_vector
    centered_norms = np.linalg.norm(centered_trajectory, axis=1, keepdims=True)
    norm_centered = centered_trajectory / (centered_norms + 1e-9)
    
    centered_consecutive_sims = []
    for t in range(T - 1):
        sim = np.dot(norm_centered[t], norm_centered[t+1])
        centered_consecutive_sims.append(float(sim))
        
    centered_init_sims = []
    for t in range(T):
        sim = np.dot(norm_centered[t], norm_centered[0])
        centered_init_sims.append(float(sim))
        
    # --- 3. PCA Projecting ---
    if T >= 2:
        pca = PCA(n_components=2)
        trajectory_2d = pca.fit_transform(trajectory)
        explained_variance = pca.explained_variance_ratio_.tolist()
    else:
        trajectory_2d = np.zeros((T, 2))
        explained_variance = [0.0, 0.0]
        
    return {
        "consecutive_sims": consecutive_sims,
        "init_sims": init_sims,
        "centered_consecutive_sims": centered_consecutive_sims,
        "centered_init_sims": centered_init_sims,
        "trajectory_2d": trajectory_2d.tolist(),
        "explained_variance": explained_variance
    }

def plot_representation_analysis(metrics, output_image_path):
    """
    Generates plots of cosine similarity over time (raw and centered) and the PCA trajectory path.
    """
    consecutive_sims = metrics["consecutive_sims"]
    init_sims = metrics["init_sims"]
    centered_consecutive_sims = metrics.get("centered_consecutive_sims", [])
    centered_init_sims = metrics.get("centered_init_sims", [])
    traj_2d = np.array(metrics["trajectory_2d"])
    T = len(init_sims)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot 1: Cosine Similarity metrics
    # Raw Similarity on primary left y-axis
    color_raw = 'purple'
    ax1 = axes[0]
    ax1.plot(range(T - 1), consecutive_sims, label='Raw Consecutive S(t, t+1)', color=color_raw, marker='o')
    ax1.plot(range(T), init_sims, label='Raw Initial S(t, 0)', color=color_raw, linestyle='--')
    ax1.set_xlabel('Temporal Patch Index (t)')
    ax1.set_ylabel('Raw Cosine Similarity', color=color_raw)
    ax1.tick_params(axis='y', labelcolor=color_raw)
    ax1.set_ylim([0.0, 1.05])
    ax1.grid(True)
    
    # Mean-centered correlation on secondary right y-axis
    if len(centered_consecutive_sims) > 0:
        color_centered = 'teal'
        ax2 = ax1.twinx()
        ax2.plot(range(T - 1), centered_consecutive_sims, label='Centered Consecutive (Correlation)', color=color_centered, marker='s', alpha=0.7)
        ax2.plot(range(T), centered_init_sims, label='Centered Initial', color=color_centered, linestyle=':', alpha=0.7)
        ax2.set_ylabel('Mean-Centered Correlation', color=color_centered)
        ax2.tick_params(axis='y', labelcolor=color_centered)
        ax2.set_ylim([-1.05, 1.05])
        
        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower left', fontsize='small')
    else:
        ax1.legend(loc='lower left')
        
    axes[0].set_title('Representational Similarity Over Time')
    
    # Plot 2: 2D PCA state space trajectory
    sc = axes[1].scatter(traj_2d[:, 0], traj_2d[:, 1], c=range(T), cmap='plasma', edgecolor='k', s=50, zorder=3)
    axes[1].plot(traj_2d[:, 0], traj_2d[:, 1], color='gray', linestyle='-', alpha=0.5, zorder=2)
    
    # Annotate start and end points
    axes[1].text(traj_2d[0, 0], traj_2d[0, 1], ' Start', color='green', fontweight='bold')
    axes[1].text(traj_2d[-1, 0], traj_2d[-1, 1], ' End', color='red', fontweight='bold')
    
    axes[1].set_xlabel('PCA Component 1 (captures temporal drift)')
    axes[1].set_ylabel('PCA Component 2 (captures state changes)')
    axes[1].set_title('2D PCA Representation Space Trajectory')
    fig.colorbar(sc, ax=axes[1], label='Time Step (t)')
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(output_image_path, dpi=300)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Run Experiment 2: Representation Similarity Analysis")
    parser.add_argument("--video-path", type=str, default=None, help="Path to a single video file")
    parser.add_argument("--video-dir", type=str, default=None, help="Path to a directory containing video dataset")
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen3-VL-8B-Instruct", help="Hugging Face model ID")
    parser.add_argument("--layer-idx", type=int, default=-2, help="Layer index to extract hidden states")
    parser.add_argument("--output-dir", type=str, default="results/exp2", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda", help="Target device")
    parser.add_argument("--prompt-mode", type=str, default="both", choices=["cot", "direct", "both"], help="Prompting mode (cot, direct, or both)")
    args = parser.parse_args()
    
    # Load model and processor once
    model, processor = load_model_and_processor(args.model_id, device=args.device)
    
    # Resolve cohorts to process
    cohorts = []
    if args.video_path:
        cohorts.append((args.video_path, True))
    elif args.video_dir:
        cohorts.append((args.video_dir, False))
    else:
        # Default to all 3 domains
        for d in ["videos/temporal/blinking", "videos/temporal/bounce_ball", "videos/temporal/state_machine"]:
            if os.path.exists(d):
                cohorts.append((d, False))
                
    if not cohorts:
        print("Error: No target video, video-dir, or default temporal video domains found.")
        return
        
    for target_path, is_single_video in cohorts:
        # Resolve domain-specific output directory
        domain_name = "blinking"
        if "bounce" in target_path.lower():
            domain_name = "bounce_ball"
        elif "state" in target_path.lower() or "transition" in target_path.lower():
            domain_name = "state_machine"
            
        # Collect video instances
        instances = []
        if is_single_video:
            instances.append((get_associated_files(target_path), "single_video"))
            print(f"\nTargeting single video in domain '{domain_name}': {target_path}")
        else:
            all_instances = find_video_files(target_path)
            print(f"\nTargeting cohort directory for domain '{domain_name}': {target_path} (Found {len(all_instances)} videos)")
            
            # Select Cohorts A, B, and C:
            # Cohort A (Easy): N <= 3, f <= 1.0
            cohort_A = [inst for inst in all_instances if inst["metadata"]["count"] is not None and inst["metadata"]["count"] <= 3 and inst["metadata"]["frequency"] is not None and inst["metadata"]["frequency"] <= 1.0]
            # Cohort B (Trap): N >= 5, f <= 1.0
            cohort_B = [inst for inst in all_instances if inst["metadata"]["count"] is not None and inst["metadata"]["count"] >= 5 and inst["metadata"]["frequency"] is not None and inst["metadata"]["frequency"] <= 1.0]
            # Cohort C (High-Freq): N >= 5, f >= 3.0
            cohort_C = [inst for inst in all_instances if inst["metadata"]["count"] is not None and inst["metadata"]["count"] >= 5 and inst["metadata"]["frequency"] is not None and inst["metadata"]["frequency"] >= 3.0]
            
            import random
            # Shuffle deterministically to get diverse seeds
            random.Random(42).shuffle(cohort_A)
            random.Random(42).shuffle(cohort_B)
            random.Random(42).shuffle(cohort_C)
            
            # Select up to 4 representative videos from each cohort
            selected_cohort_A = cohort_A[:4]
            selected_cohort_B = cohort_B[:4]
            selected_cohort_C = cohort_C[:4]
            
            for inst in selected_cohort_A:
                instances.append((inst, "cohort_A"))
            for inst in selected_cohort_B:
                instances.append((inst, "cohort_B"))
            for inst in selected_cohort_C:
                instances.append((inst, "cohort_C"))
            
            print(f"  Selected {len(selected_cohort_A)} videos for Cohort A (Easy)")
            print(f"  Selected {len(selected_cohort_B)} videos for Cohort B (Trap)")
            print(f"  Selected {len(selected_cohort_C)} videos for Cohort C (High-Freq)")

        # Loop over prompt modes
        modes_to_run = ["cot", "direct"] if args.prompt_mode == "both" else [args.prompt_mode]
        for prompt_mode in modes_to_run:
            print(f"\n--- Running Prompt Mode: {prompt_mode.upper()} ---")
            output_dir = os.path.join(args.output_dir, domain_name, prompt_mode)
            os.makedirs(output_dir, exist_ok=True)
            
            all_metrics = []
            
            for idx, (inst, cohort_label) in enumerate(instances):
                video_path = inst["video_path"]
                q_path = inst["question_path"]
                metadata = inst["metadata"]
                
                print(f"  Processing [{idx+1}/{len(instances)}] ({cohort_label}): {os.path.basename(video_path)}")
                
                if not q_path:
                    raw_question = "How many times did the object flash?"
                    print(f"    Warning: Question file not found. Using default question.")
                else:
                    with open(q_path, "r") as f:
                        raw_question = f.read().strip()
                        
                question_text = format_prompt_by_mode(raw_question, prompt_mode)
                inputs = prepare_video_inputs(video_path, question_text, processor, device=args.device)
                
                try:
                    # Extract representations
                    trajectory = extract_representation_trajectory(model, inputs, processor, layer_idx=args.layer_idx)
                    
                    # Compute similarities & PCA coordinates
                    metrics = compute_trajectory_metrics(trajectory)
                    metrics["metadata"] = metadata
                    metrics["video_name"] = os.path.basename(video_path)
                    metrics["cohort"] = cohort_label
                    metrics["prompt"] = question_text
                    
                    # Run text generation to verify what the model actually outputs
                    print("    Generating model's reasoning and count answer...")
                    with torch.no_grad():
                        generated_ids = model.generate(**inputs, max_new_tokens=512)
                        generated_ids = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)]
                        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                    metrics["generated_response"] = generated_text
                    print(f"    Generated Response: {generated_text.strip().replace(chr(10), ' ')}")
                    
                    # Save visual plots for each video with cohort label in filename
                    out_img = os.path.join(output_dir, f"{cohort_label}_{os.path.splitext(os.path.basename(video_path))[0]}_repr.png")
                    plot_representation_analysis(metrics, out_img)
                    
                    all_metrics.append(metrics)
                except Exception as e:
                    print(f"    Error processing {video_path}: {e}")
                finally:
                    # Aggressive cleanup of input and generation tensors
                    if "inputs" in locals() and inputs is not None:
                        for k in list(inputs.keys()):
                            inputs[k] = None
                        inputs = None
                    if "trajectory" in locals():
                        trajectory = None
                    if "generated_ids" in locals():
                        generated_ids = None
                    import gc
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
            if not all_metrics:
                continue
                
            # Save results summary to JSON
            out_json = os.path.join(output_dir, "representation_similarity_summary.json")
            with open(out_json, "w") as f:
                json.dump(all_metrics, f, indent=2)
                
            print(f"Finished representation similarity analysis for {domain_name} ({prompt_mode})! Summary saved to {out_json}")

if __name__ == "__main__":
    main()
