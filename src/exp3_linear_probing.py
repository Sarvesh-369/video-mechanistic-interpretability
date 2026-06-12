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
    Parses a reasoning trace file to determine the ground-truth state (ON=1, OFF=0) of the object at each temporal step.
    """
    with open(trace_path, "r") as f:
        content = f.read()
        
    # Find all event lines: "At M:SS.CSs: Object ..."
    event_lines = re.findall(r'- At (\d+:\d+\.\d+)s: Object ([^\n]+)', content)
    
    events = []
    for time_str, desc in event_lines:
        parts = time_str.split(":")
        minutes = float(parts[0])
        seconds = float(parts[1])
        timestamp = minutes * 60.0 + seconds
        
        state = None
        if "appears" in desc:
            state = 1 if "(ON state)" in desc else 0
        elif "turns ON" in desc:
            state = 1
        elif "turns OFF" in desc:
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
        
    return np.array(labels)

def collect_features_and_labels(model, processor, instances, layer_idx, device):
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
            
            labels = parse_trace_states(trace_path, T, duration)
            
            X_list.append(trajectory)
            y_list.append(labels)
            metadata_list.append(meta)
            video_names.append(os.path.basename(video_path))
            
            print(f"  Loaded [{idx+1}/{len(instances)}]: {os.path.basename(video_path)} (T={T})")
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
    parser.add_argument("--train-dir", type=str, required=True, help="Path to the training videos task directory (e.g. videos/temporal/blinking)")
    parser.add_argument("--video-path", type=str, default=None, help="Path to a single test video file")
    parser.add_argument("--test-dir", type=str, default=None, help="Path to a directory containing test videos")
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen3-VL-8B-Instruct", help="Hugging Face model ID")
    parser.add_argument("--layer-idx", type=int, default=-2, help="Layer index to probe")
    parser.add_argument("--output-dir", type=str, default="results/exp3", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda", help="Target device")
    parser.add_argument("--max-train-videos", type=int, default=50, help="Maximum number of training videos to sample")
    parser.add_argument("--c", type=float, default=0.1, help="Inverse of regularization strength for Logistic Regression")
    args = parser.parse_args()
    
    if (args.video_path is None) == (args.test_dir is None):
        parser.error("Exactly one of --video-path or --test-dir must be provided for evaluation.")
        
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model and processor
    model, processor = load_model_and_processor(args.model_id, device=args.device)
    
    # 1. Collect training data and train the probe
    print("\n--- 1. Collecting Probing Training Data (Event Count <= 3) ---")
    train_instances = find_video_files(args.train_dir)
    easy_train_instances = [inst for inst in train_instances if inst["metadata"]["count"] is not None and inst["metadata"]["count"] <= 3]
    
    # Shuffle deterministically to get a diverse mix of event counts (c0, c1, c2, c3)
    # instead of just taking the first alphabetically sorted ones (which are all c0)
    shuffled_train = list(easy_train_instances)
    random.Random(42).shuffle(shuffled_train)
    easy_train_instances = shuffled_train[:args.max_train_videos]
    
    X_train_list, y_train_list, _, _ = collect_features_and_labels(
        model, processor, easy_train_instances, args.layer_idx, args.device
    )
    
    if not X_train_list:
        print("Error: No training representations collected.")
        return
        
    X_train = np.concatenate(X_train_list, axis=0)
    y_train = np.concatenate(y_train_list, axis=0)
    
    print(f"\nTraining Logistic Regression probe on {X_train.shape[0]} training frame representations (C={args.c})...")
    probe = LogisticRegression(max_iter=1000, C=args.c)
    probe.fit(X_train, y_train)
    
    train_acc = accuracy_score(y_train, probe.predict(X_train))
    print(f"Probe Train Accuracy: {train_acc:.4f}")
    
    # Save the trained linear probe model to be used later
    probe_path = os.path.join(args.output_dir, "linear_probe_model.pkl")
    with open(probe_path, "wb") as f:
        pickle.dump(probe, f)
    print(f"Saved trained linear probe model to {probe_path}")
    
    # 2. Collect evaluation data
    print("\n--- 2. Collecting Evaluation Data ---")
    eval_instances = []
    if args.video_path:
        eval_instances.append(get_associated_files(args.video_path))
        print(f"Targeting single test video: {args.video_path}")
    else:
        # Find test instances with event count >= 5
        test_instances = find_video_files(args.test_dir)
        hard_test_instances = [inst for inst in test_instances if inst["metadata"]["count"] is not None and inst["metadata"]["count"] >= 5]
        
        # Shuffle deterministically to get a diverse mix of test instances
        shuffled_test = list(hard_test_instances)
        random.Random(42).shuffle(shuffled_test)
        eval_instances = shuffled_test[:15]
        print(f"Targeting test directory: {args.test_dir} (Selected {len(eval_instances)} diverse hard instances)")
        
    X_eval_list, y_eval_list, eval_meta, eval_names = collect_features_and_labels(
        model, processor, eval_instances, args.layer_idx, args.device
    )
    
    if not X_eval_list:
        print("Error: No evaluation representations collected.")
        return
        
    # Evaluate probe
    X_eval = np.concatenate(X_eval_list, axis=0)
    y_eval = np.concatenate(y_eval_list, axis=0)
    y_eval_pred = probe.predict(X_eval)
    eval_acc = accuracy_score(y_eval, y_eval_pred)
    
    print("\n--- Probing Evaluation Results ---")
    print(f"Probing Accuracy: {eval_acc:.4f}")
    
    # Compute classification report string and dict (with zero_division=0 to prevent terminal warnings)
    report_str = classification_report(y_eval, y_eval_pred, labels=[0, 1], target_names=["OFF", "ON"], zero_division=0)
    report_dict = classification_report(y_eval, y_eval_pred, labels=[0, 1], target_names=["OFF", "ON"], output_dict=True, zero_division=0)
    print(report_str)
    
    # Save individual prediction plots
    print("\nSaving predicted state trajectory plots...")
    for i in range(min(3, len(X_eval_list))):
        y_true_v = y_eval_list[i]
        X_v = X_eval_list[i]
        y_pred_v = probe.predict(X_v)
        
        v_name = eval_names[i]
        out_img = os.path.join(args.output_dir, f"{os.path.splitext(v_name)[0]}_probe.png")
        
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
    out_json = os.path.join(args.output_dir, "probing_evaluation_results.json")
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nFinished Experiment 3! Summary saved to {out_json}")

if __name__ == "__main__":
    main()
