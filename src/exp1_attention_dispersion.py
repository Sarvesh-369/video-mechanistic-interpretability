import os
import argparse
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from src.utils.model_helpers import load_model_and_processor, prepare_video_inputs, get_associated_files, find_video_files

def extract_temporal_attention(model, inputs, processor, query_token_pos=-1):
    """
    Extracts the attention weights from a query token back to the video frames.
    
    Args:
        model: Loaded Qwen3-VL model.
        inputs: Tokenized input dictionary containing pixel_values_videos, video_grid_thw, etc.
        processor: Loaded AutoProcessor.
        query_token_pos (int): Index of the query token (default is -1, the last token in prompt/input).
    
    Returns:
        dict: Containing layer-wise temporal attention distributions and computed entropy metrics.
    """
    device = next(model.parameters()).device
    
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
    
    input_ids = inputs["input_ids"][0]
    
    # Locate visual tokens
    vocab = processor.tokenizer.get_vocab()
    vision_start_id = vocab.get("<|vision_start|>", 151652)
    vision_end_id = vocab.get("<|vision_end|>", 151653)
    
    start_pos = (input_ids == vision_start_id).nonzero(as_tuple=True)[0]
    end_pos = (input_ids == vision_end_id).nonzero(as_tuple=True)[0]
    
    if len(start_pos) == 0 or len(end_pos) == 0:
        raise ValueError("Could not locate <|vision_start|> or <|vision_end|> tokens.")
        
    start_idx = start_pos[0].item() + 1
    end_idx = end_pos[0].item()
    num_vision_tokens = end_idx - start_idx
    
    # Get grid information
    if "video_grid_thw" not in inputs:
        raise ValueError("video_grid_thw not found in inputs.")
        
    video_grid_thw = inputs["video_grid_thw"][0].cpu().numpy()
    T, H, W = int(video_grid_thw[0]), int(video_grid_thw[1]), int(video_grid_thw[2])
    
    # Extract attention maps
    num_layers = len(outputs.attentions)
    num_heads = outputs.attentions[0].shape[1]
    
    layer_entropies = []
    layer_attentions = []
    
    for layer in range(num_layers):
        attn_matrix = outputs.attentions[layer][0].cpu().numpy() # shape: (num_heads, seq_len, seq_len)
        
        head_entropies = []
        head_attns = []
        
        for head in range(num_heads):
            # Extract attention weights
            query_to_vision = attn_matrix[head, query_token_pos, start_idx:end_idx]
            
            temporal_attn = np.zeros(T)
            patches_per_frame = H * W
            
            for t in range(T):
                start_p = t * patches_per_frame
                end_p = (t + 1) * patches_per_frame
                if end_p <= len(query_to_vision):
                    temporal_attn[t] = np.sum(query_to_vision[start_p:end_p])
            
            # Normalize
            sum_val = np.sum(temporal_attn)
            if sum_val > 0:
                temporal_attn_norm = temporal_attn / sum_val
            else:
                temporal_attn_norm = np.ones(T) / T
                
            # Compute Shannon entropy
            eps = 1e-9
            entropy = -np.sum(temporal_attn_norm * np.log(temporal_attn_norm + eps))
            
            head_entropies.append(float(entropy))
            head_attns.append(temporal_attn_norm.tolist())
            
        layer_entropies.append(head_entropies)
        layer_attentions.append(head_attns)
        
    return {
        "num_layers": num_layers,
        "num_heads": num_heads,
        "T": int(T),
        "layer_entropies": layer_entropies, # shape: (layers, heads)
        "layer_attentions": layer_attentions, # shape: (layers, heads, T)
        "max_possible_entropy": float(np.log(T))
    }

def plot_attention_dispersion(results, output_image_path):
    """
    Plots attention entropy across layers and generates a temporal attention heatmap.
    """
    layers = results["num_layers"]
    T = results["T"]
    
    entropies = np.array(results["layer_entropies"])
    avg_entropies = np.mean(entropies, axis=1)
    max_entropy = results["max_possible_entropy"]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Entropy across layers
    axes[0].plot(range(layers), avg_entropies, marker='o', color='b', label='Avg Head Entropy')
    axes[0].axhline(y=max_entropy, color='r', linestyle='--', label='Max Entropy (Uniform)')
    axes[0].set_xlabel('Layer')
    axes[0].set_ylabel('Temporal Attention Entropy (Nats)')
    axes[0].set_title(f"Attention Dispersion per Layer (T={T})")
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot 2: Heatmap
    attns = np.array(results["layer_attentions"])
    avg_attns = np.mean(attns, axis=1)
    
    im = axes[1].imshow(avg_attns, aspect='auto', cmap='viridis', origin='lower')
    axes[1].set_xlabel('Video Time (Temporal Patches)')
    axes[1].set_ylabel('Layer')
    axes[1].set_title('Average Temporal Attention Weight')
    fig.colorbar(im, ax=axes[1], label='Attention Weight')
    
    plt.tight_layout()
    plt.savefig(output_image_path, dpi=300)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Run Experiment 1: Attention Dispersion Analysis")
    parser.add_argument("--video-path", type=str, default=None, help="Path to a single video file")
    parser.add_argument("--video-dir", type=str, default=None, help="Path to a directory containing video dataset")
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen3-VL-8B-Instruct", help="Hugging Face model ID")
    parser.add_argument("--output-dir", type=str, default="results/exp1", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda", help="Target device")
    args = parser.parse_args()
    
    if (args.video_path is None) == (args.video_dir is None):
        parser.error("Exactly one of --video-path or --video-dir must be provided.")
        
    # Resolve domain-specific output directory
    domain_name = "blinking"
    target_path = args.video_path or args.video_dir
    if target_path:
        if "bounce" in target_path.lower():
            domain_name = "bounce_ball"
        elif "state" in target_path.lower() or "transition" in target_path.lower():
            domain_name = "state_machine"
            
    output_dir = os.path.join(args.output_dir, domain_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model and processor
    model, processor = load_model_and_processor(args.model_id, device=args.device)
    
    # Collect video instances
    instances = []
    if args.video_path:
        instances.append(get_associated_files(args.video_path))
        print(f"Targeting single video: {args.video_path}")
    else:
        instances = find_video_files(args.video_dir)
        print(f"Targeting cohort directory: {args.video_dir} (Found {len(instances)} videos)")
        # Limit to a subset of 10 to keep runtimes reasonable in directories
        instances = instances[:10]
        
    all_results = []
    
    for idx, inst in enumerate(instances):
        video_path = inst["video_path"]
        q_path = inst["question_path"]
        metadata = inst["metadata"]
        
        print(f"\nProcessing [{idx+1}/{len(instances)}]: {os.path.basename(video_path)}")
        
        if not q_path:
            question_text = "How many times did the object flash? Show your reasoning and put the final answer in \\boxed{}"
            print(f"  Warning: Question file not found. Using default question: '{question_text}'")
        else:
            with open(q_path, "r") as f:
                question_text = f.read().strip()
                
        inputs = prepare_video_inputs(video_path, question_text, processor, device=args.device)
        
        try:
            results = extract_temporal_attention(model, inputs, processor, query_token_pos=-1)
            results["metadata"] = metadata
            results["video_name"] = os.path.basename(video_path)
            
            # Save visual plots for each video
            out_img = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(video_path))[0]}_attn.png")
            plot_attention_dispersion(results, out_img)
            
            all_results.append(results)
        except Exception as e:
            print(f"  Error processing {video_path}: {e}")
            
    # Save results summary to JSON
    out_json = os.path.join(output_dir, "attention_dispersion_summary.json")
    summary_results = []
    for r in all_results:
        summary_results.append({
            "video_name": r["video_name"],
            "metadata": r["metadata"],
            "layer_entropies": r["layer_entropies"],
            "max_possible_entropy": r["max_possible_entropy"]
        })
        
    with open(out_json, "w") as f:
        json.dump(summary_results, f, indent=2)
        
    print(f"\nFinished Experiment 1! Results written to {out_json}")

if __name__ == "__main__":
    main()
