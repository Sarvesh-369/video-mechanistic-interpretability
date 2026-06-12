import os
import argparse
import json
import torch
import torch.nn.functional as F
from src.utils.model_helpers import load_model_and_processor, prepare_video_inputs, get_associated_files, find_video_files

def run_logit_lens(model, inputs, processor, correct_token_str, alternative_token_strs):
    """
    Performs logit lens projection on the last query token in inputs across all layers.
    """
    tokenizer = processor.tokenizer
    
    correct_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(correct_token_str)[-1])
    alt_ids = {
        token_str: tokenizer.convert_tokens_to_ids(tokenizer.tokenize(token_str)[-1])
        for token_str in alternative_token_strs
    }
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        
    num_layers = len(outputs.hidden_states) - 1
    query_pos = -1
    
    if hasattr(model, "model") and hasattr(model.model, "norm"):
        final_norm = model.model.norm
    elif hasattr(model, "norm"):
        final_norm = model.norm
    else:
        final_norm = lambda x: x
        
    lm_head = model.lm_head
    
    layer_probs_correct = []
    layer_probs_alts = {token_str: [] for token_str in alternative_token_strs}
    
    for layer in range(num_layers + 1):
        h_L = outputs.hidden_states[layer][0, query_pos]
        
        with torch.no_grad():
            normed_h = final_norm(h_L.unsqueeze(0))
            logits = lm_head(normed_h)[0]
            probs = F.softmax(logits, dim=-1).cpu().numpy()
            
        layer_probs_correct.append(float(probs[correct_id]))
        for token_str, token_id in alt_ids.items():
            layer_probs_alts[token_str].append(float(probs[token_id]))
            
    return {
        "num_layers": num_layers,
        "correct_token": correct_token_str,
        "correct_probs": layer_probs_correct,
        "alternative_probs": layer_probs_alts
    }

def plot_logit_lens(results, output_image_path, title):
    """
    Plots the logit lens probabilities across the model's layers.
    """
    layers = results["num_layers"]
    correct_token = results["correct_token"]
    correct_probs = results["correct_probs"]
    alt_probs = results["alternative_probs"]
    
    plt.figure(figsize=(10, 5))
    
    # Plot correct token probability
    plt.plot(range(layers + 1), correct_probs, label=f"Correct ('{correct_token}')", color='green', linewidth=2.5, marker='o')
    
    # Plot alternative tokens probabilities
    colors = ['orange', 'red', 'blue', 'purple']
    for idx, (token_str, probs) in enumerate(alt_probs.items()):
        color = colors[idx % len(colors)]
        plt.plot(range(layers + 1), probs, label=f"Alternative ('{token_str}')", color=color, linestyle='--', marker='x')
        
    plt.xlabel('Layer Index')
    plt.ylabel('Vocabulary Probability')
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_image_path, dpi=300)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Run Experiment 5: Logit Lens Analysis")
    parser.add_argument("--video-path", type=str, default=None, help="Path to a single video file")
    parser.add_argument("--video-dir", type=str, default=None, help="Path to a directory containing video dataset")
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen3-VL-8B-Instruct", help="Hugging Face model ID")
    parser.add_argument("--output-dir", type=str, default="results/exp5", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda", help="Target device")
    args = parser.parse_args()
    
    if (args.video_path is None) == (args.video_dir is None):
        parser.error("Exactly one of --video-path or --video-dir must be provided.")
        
    # Import matplotlib here
    global plt
    import matplotlib.pyplot as plt
    
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
            
        output_dir = os.path.join(args.output_dir, domain_name)
        os.makedirs(output_dir, exist_ok=True)
        
        # Resolve target video
        if is_single_video:
            inst = get_associated_files(target_path)
            print(f"\nTargeting single video in domain '{domain_name}': {target_path}")
        else:
            # Select first failing video from directory
            all_instances = find_video_files(target_path)
            failing_instances = [i for i in all_instances if i["metadata"]["count"] is not None and i["metadata"]["count"] >= 5]
            if not failing_instances:
                failing_instances = all_instances
                
            if not failing_instances:
                print(f"Error: No videos found in directory {target_path}.")
                continue
                
            inst = failing_instances[0]
            print(f"\nTargeting first failing video from directory for domain '{domain_name}': {inst['video_path']}")
            
        video_path = inst["video_path"]
        q_path = inst["question_path"]
        solution_path = inst["solution_path"]
        metadata = inst["metadata"]
        
        if not q_path or not solution_path:
            print(f"Warning: Associated question or solution file not found for {video_path}")
            continue
            
        with open(q_path, "r") as f:
            question_text = f.read().strip()
            
        with open(solution_path, "r") as f:
            ground_truth = int(f.read().strip())
            
        print(f"  Running Logit Lens on: {os.path.basename(video_path)} (GT Count = {ground_truth})")
        
        inputs = prepare_video_inputs(video_path, question_text, processor, device=args.device)
        
        correct_token_str = str(ground_truth)
        alternative_token_strs = [str(ground_truth - 1), str(ground_truth - 2), str(ground_truth + 1)]
        
        try:
            results = run_logit_lens(model, inputs, processor, correct_token_str, alternative_token_strs)
            results["video_name"] = os.path.basename(video_path)
            
            # Save Logit Lens Plot
            out_img = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(video_path))[0]}_logit_lens.png")
            plot_logit_lens(
                results, out_img, 
                f"Logit Lens Profile for {os.path.basename(video_path)} (GT={ground_truth})"
            )
            
            # Save JSON output
            out_json = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(video_path))[0]}_logit_lens.json")
            with open(out_json, "w") as f:
                json.dump(results, f, indent=2)
            print(f"Finished Experiment 5 for {domain_name}! Summary saved to {out_json}")
        except Exception as e:
            print(f"  Error running Logit Lens: {e}")

if __name__ == "__main__":
    main()
