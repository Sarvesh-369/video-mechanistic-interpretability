import os
import argparse
import re
import json
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
from src.utils.model_helpers import (
    load_model_and_processor, 
    prepare_video_inputs, 
    find_video_files, 
    get_associated_files, 
    extract_representation_trajectory
)

def parse_trace_states(trace_path, T, duration):
    """
    Parses a reasoning trace file to determine the ground-truth state (1 or 0)
    of the object/ball/circle at each temporal step.
    Supports all 3 domains: blinking, bounce_ball, and state_machine.
    """
    with open(trace_path, "r") as f:
        lines = f.readlines()
        
    events = []
    
    # Identify domain from file path or content
    is_bounce = "bounce" in trace_path.lower()
    is_state_machine = "state" in trace_path.lower() or "transition" in trace_path.lower()
    
    for line in lines:
        line = line.strip()
        # Parse timestamp from format like "At M:SS.CSs" or "At S.CSs"
        # e.g., "- At 0:00.00s: ..." or "At 0:00.83s, ..."
        time_match = re.search(r'At (\d+:\d+\.\d+|\d+\.\d+)s', line)
        if not time_match:
            continue
            
        time_str = time_match.group(1)
        if ":" in time_str:
            parts = time_str.split(":")
            minutes = float(parts[0])
            seconds = float(parts[1])
            timestamp = minutes * 60.0 + seconds
        else:
            timestamp = float(time_str)
            
        state = None
        if is_bounce:
            # Bounce Ball domain: Wall B (Positive) vs Wall A (Negative)
            if "Wall B" in line or "Positive" in line:
                state = 1
            elif "Wall A" in line or "Negative" in line:
                state = 0
            elif "appears" in line:
                # e.g. "Ball appears at x=1.17 (local)"
                x_match = re.search(r'x=(-?\d+\.?\d*)', line)
                if x_match:
                    state = 1 if float(x_match.group(1)) > 0 else 0
                else:
                    state = 1
        elif is_state_machine:
            # State Machine domain: color mapping
            # Primary split: warm colors (RED, YELLOW) = 1, cool/others (GREEN, BLUE) = 0
            for c in ["RED", "YELLOW"]:
                if c in line:
                    state = 1
                    break
            for c in ["GREEN", "BLUE"]:
                if c in line:
                    state = 0
                    break
        else:
            # Blinking domain: ON vs OFF
            if "appears" in line:
                state = 1 if "(ON state)" in line else 0
            elif "turns ON" in line:
                state = 1
            elif "turns OFF" in line:
                state = 0
                
        if state is not None:
            events.append((timestamp, state))
            
    events = sorted(events, key=lambda x: x[0])
    
    labels = []
    for t in range(T):
        time_step = t * (duration / T)
        
        active_state = events[0][1] if events else 0
        for event_time, event_state in events:
            if time_step >= event_time:
                active_state = event_state
            else:
                break
        labels.append(active_state)
        
    last_event_time = events[-1][0] if events else 0.0
    return np.array(labels), last_event_time

def collect_features_and_labels(model, processor, instances, layer_idx, device, crop_active_only=False, crop_buffer=1.0):
    """
    Extracts representations and parses labels for a set of instances.
    """
    X_list = []
    y_list = []
    metadata_list = []
    video_names = []
    
    for idx, inst in enumerate(instances):
        video_path = inst["video_path"]
        q_path = inst["question_path"]
        trace_path = inst["trace_path"]
        meta = inst["metadata"]
        
        if not trace_path or not q_path:
            continue
            
        with open(q_path, "r") as f:
            question_text = f.read().strip()
            
        inputs = prepare_video_inputs(video_path, question_text, processor, device=device)
        
        try:
            trajectory = extract_representation_trajectory(model, inputs, processor, layer_idx=layer_idx)
            T = trajectory.shape[0]
            duration = meta["duration"]
            
            labels, last_event_time = parse_trace_states(trace_path, T, duration)
            
            if crop_active_only:
                crop_limit = max(4.0, last_event_time + crop_buffer)
                valid_indices = [t for t in range(T) if t * (duration / T) <= crop_limit]
                if not valid_indices:
                    valid_indices = [0]
                trajectory = trajectory[valid_indices]
                labels = labels[valid_indices]
                T = len(valid_indices)
            
            X_list.append(trajectory)
            y_list.append(labels)
            metadata_list.append(meta)
            video_names.append(os.path.basename(video_path))
            
            crop_str = f" (cropped T={T})" if crop_active_only else ""
            print(f"  Loaded [{idx+1}/{len(instances)}]: {os.path.basename(video_path)}{crop_str}")
        except Exception as e:
            print(f"  Error loading {video_path}: {e}")
            
    return X_list, y_list, metadata_list, video_names

def plot_probe_predictions(y_true, y_pred, output_image_path, title):
    """
    Plots the true state trajectory vs. probe predicted state trajectory.
    """
    T = len(y_true)
    plt.figure(figsize=(10, 3))
    plt.step(range(T), y_true, where='post', label='Ground Truth State', color='green', linewidth=2)
    plt.step(range(T), y_pred, where='post', label='Linear Probe Predicted', color='orange', linestyle='--', linewidth=2)
    plt.xlabel('Time Step (t)')
    plt.ylabel('State (ON=1, OFF=0)')
    plt.title(title)
    plt.ylim([-0.2, 1.2])
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_image_path, dpi=300)
    plt.close()

def main():
    import random
    
    parser = argparse.ArgumentParser(description="Run Experiment 3: Linear Probing for Perceptual State Preservation")
    parser.add_argument("--train-dir", type=str, default=None, help="Path to the training videos task directory (e.g. videos/temporal/blinking)")
    parser.add_argument("--video-path", type=str, default=None, help="Path to a single test video file")
    parser.add_argument("--test-dir", type=str, default=None, help="Path to a directory containing test videos")
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen3-VL-8B-Instruct", help="Hugging Face model ID")
    parser.add_argument("--layer-idx", type=int, default=-2, help="Layer index to probe")
    parser.add_argument("--output-dir", type=str, default="results/exp3", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda", help="Target device")
    parser.add_argument("--max-train-videos", type=int, default=100, help="Maximum number of training videos to sample")
    parser.add_argument("--regularization-c", type=float, default=0.1, help="Inverse of regularization strength for Logistic Regression")
    parser.add_argument("--no-crop-active", action="store_true", help="Disable temporal cropping of active events (use full video duration)")
    parser.add_argument("--crop-buffer", type=float, default=1.0, help="Buffer in seconds to append after the last event timestamp when cropping")
    parser.add_argument("--no-class-balance", action="store_true", help="Disable balanced class weighting in Logistic Regression probe")
    args = parser.parse_args()
    
    # Load model and processor once
    model, processor = load_model_and_processor(args.model_id, device=args.device)
    
    # Resolve cohorts of train/eval paths
    cohorts = []
    if args.video_path:
        if not args.train_dir:
            parser.error("--train-dir must be specified when using --video-path")
        cohorts.append((args.train_dir, None, args.video_path))
    elif args.test_dir:
        if not args.train_dir:
            parser.error("--train-dir must be specified when using --test-dir")
        cohorts.append((args.train_dir, args.test_dir, None))
    else:
        # Default to all 3 domains
        for d in ["videos/temporal/blinking", "videos/temporal/bounce_ball", "videos/temporal/state_machine"]:
            if os.path.exists(d):
                cohorts.append((d, d, None))
                
    if not cohorts:
        print("Error: No training and evaluation configurations resolved.")
        return
        
    for train_dir, test_dir, video_path in cohorts:
        # Resolve domain name
        domain_name = "blinking"
        target_path = train_dir or video_path
        if target_path:
            if "bounce" in target_path.lower():
                domain_name = "bounce_ball"
            elif "state" in target_path.lower() or "transition" in target_path.lower():
                domain_name = "state_machine"
                
        output_dir = os.path.join(args.output_dir, domain_name)
        os.makedirs(output_dir, exist_ok=True)
        
        if domain_name == "bounce_ball":
            target_names = ["Wall A (Negative)", "Wall B (Positive)"]
        elif domain_name == "state_machine":
            target_names = ["Cool (GREEN/BLUE)", "Warm (RED/YELLOW)"]
        else:
            target_names = ["OFF", "ON"]
            
        # 1. Collect training data and train the probe
        print(f"\n--- 1. Collecting Probing Training Data for '{domain_name}' (Event Count <= 3) ---")
        train_instances = find_video_files(train_dir)
        easy_train_instances = [inst for inst in train_instances if inst["metadata"]["count"] is not None and inst["metadata"]["count"] <= 3]
        
        # Shuffle deterministically to get a diverse mix of event counts
        shuffled_train = list(easy_train_instances)
        random.Random(42).shuffle(shuffled_train)
        easy_train_instances = shuffled_train[:args.max_train_videos]
        
        X_train_list, y_train_list, _, _ = collect_features_and_labels(
            model, processor, easy_train_instances, args.layer_idx, args.device,
            crop_active_only=not args.no_crop_active, crop_buffer=args.crop_buffer
        )
        
        if not X_train_list:
            print(f"Error: No training representations collected for {domain_name}.")
            continue
            
        X_train = np.concatenate(X_train_list, axis=0)
        y_train = np.concatenate(y_train_list, axis=0)
        
        class_weight = None if args.no_class_balance else "balanced"
        print(f"\nTraining Logistic Regression probe on {X_train.shape[0]} training frame representations (C={args.regularization_c}, class_weight={class_weight})...")
        probe = LogisticRegression(max_iter=1000, C=args.regularization_c, class_weight=class_weight)
        probe.fit(X_train, y_train)
        
        train_acc = accuracy_score(y_train, probe.predict(X_train))
        print(f"Probe Train Accuracy: {train_acc:.4f}")
        
        # Save the trained linear probe model to be used later
        probe_path = os.path.join(output_dir, "linear_probe_model.pkl")
        with open(probe_path, "wb") as f:
            pickle.dump(probe, f)
        print(f"Saved trained linear probe model to {probe_path}")
        
        # 2. Collect evaluation data
        print(f"\n--- 2. Collecting Evaluation Data for '{domain_name}' ---")
        eval_instances = []
        if video_path:
            eval_instances.append(get_associated_files(video_path))
            print(f"Targeting single test video: {video_path}")
        else:
            # Find test instances with event count >= 5
            test_instances = find_video_files(test_dir)
            hard_test_instances = [inst for inst in test_instances if inst["metadata"]["count"] is not None and inst["metadata"]["count"] >= 5]
            
            # Shuffle deterministically to get a diverse mix of test instances
            shuffled_test = list(hard_test_instances)
            random.Random(42).shuffle(shuffled_test)
            eval_instances = shuffled_test[:15]
            print(f"Targeting test directory: {test_dir} (Selected {len(eval_instances)} diverse hard instances)")
            
        X_eval_list, y_eval_list, eval_meta, eval_names = collect_features_and_labels(
            model, processor, eval_instances, args.layer_idx, args.device,
            crop_active_only=not args.no_crop_active, crop_buffer=args.crop_buffer
        )
        
        if not X_eval_list:
            print(f"Error: No evaluation representations collected for {domain_name}.")
            continue
            
        # Evaluate probe
        X_eval = np.concatenate(X_eval_list, axis=0)
        y_eval = np.concatenate(y_eval_list, axis=0)
        y_eval_pred = probe.predict(X_eval)
        eval_acc = accuracy_score(y_eval, y_eval_pred)
        
        print(f"\n--- Probing Evaluation Results for '{domain_name}' ---")
        print(f"Probing Accuracy: {eval_acc:.4f}")
        
        # Compute classification report string and dict (with zero_division=0 to prevent terminal warnings)
        report_str = classification_report(y_eval, y_eval_pred, labels=[0, 1], target_names=target_names, zero_division=0)
        report_dict = classification_report(y_eval, y_eval_pred, labels=[0, 1], target_names=target_names, output_dict=True, zero_division=0)
        print(report_str)
        
        # Save individual prediction plots
        print("\nSaving predicted state trajectory plots...")
        for i in range(min(3, len(X_eval_list))):
            y_true_v = y_eval_list[i]
            X_v = X_eval_list[i]
            y_pred_v = probe.predict(X_v)
            
            v_name = eval_names[i]
            out_img = os.path.join(output_dir, f"{os.path.splitext(v_name)[0]}_probe.png")
            
            plot_probe_predictions(
                y_true_v, y_pred_v, out_img,
                f"Probing Predictions for {v_name} (GT Events={eval_meta[i]['count']})"
            )
            
        # Save quantitative report
        report = {
            "train_accuracy": float(train_acc),
            "probing_accuracy": float(eval_acc),
            "model_id": args.model_id,
            "layer_idx": args.layer_idx,
            "classification_report": report_dict
        }
        out_json = os.path.join(output_dir, "probing_evaluation_results.json")
        with open(out_json, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Finished Experiment 3 for {domain_name}! Summary saved to {out_json}")

if __name__ == "__main__":
    main()
