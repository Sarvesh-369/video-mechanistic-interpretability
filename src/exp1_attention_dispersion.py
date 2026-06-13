import os
import argparse
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from src.utils.model_helpers import load_model_and_processor, prepare_video_inputs, get_associated_files, find_video_files, format_prompt_by_mode

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
    
    # Register forward hooks on attention modules to slice attention weights on-the-fly.
    # This prevents storing 40 layers of (seq_len, seq_len) attention matrices in GPU memory.
    captured_attentions = {}
    hooks = []
    outputs = None
    
    try:
        def get_hook_fn(layer_idx):
            def hook_fn(module, input_args, output):
                # output is a tuple: (attn_output, attn_weights, present_key_value)
                # or (attn_output, attn_weights) depending on config
                if isinstance(output, tuple) and len(output) > 1 and output[1] is not None:
                    attn_weights = output[1]
                    # Slice: shape (batch_size, num_heads, seq_len, seq_len) -> (batch_size, num_heads, num_vision_tokens)
                    sliced = attn_weights[:, :, query_token_pos, start_idx:end_idx].detach().clone().cpu()
                    captured_attentions[layer_idx] = sliced
                    
                    # Replace the huge tensor with a small dummy tensor to free VRAM
                    new_output = list(output)
                    new_output[1] = torch.zeros(1, 1, 1, 1, device=attn_weights.device, dtype=attn_weights.dtype)
                    return tuple(new_output)
                return output
            return hook_fn

        # Find attention layers dynamically
        attn_modules = []
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            layers = model.model.layers
        elif hasattr(model, "layers"):
            layers = model.layers
        else:
            layers = None
            
        if layers is not None:
            for idx, layer in enumerate(layers):
                if hasattr(layer, "self_attn"):
                    attn_modules.append((idx, layer.self_attn))
        else:
            # Fallback recursive search
            idx = 0
            for name, module in model.named_modules():
                if name.endswith(".self_attn"):
                    attn_modules.append((idx, module))
                    idx += 1
                    
        for layer_idx, module in attn_modules:
            hook = module.register_forward_hook(get_hook_fn(layer_idx))
            hooks.append(hook)
            
        try:
            with torch.no_grad():
                outputs = model(**inputs, output_attentions=True)
        finally:
            # Ensure hooks are always removed
            for hook in hooks:
                hook.remove()
                
        num_layers = len(attn_modules)
        
        # Safe getter for number of heads dynamically from captured attention shape, with safe configs fallbacks
        if num_layers > 0 and 0 in captured_attentions:
            num_heads = captured_attentions[0].shape[1]
        elif hasattr(model.config, "text_config"):
            num_heads = getattr(model.config.text_config, "num_attention_heads", 32)
        else:
            num_heads = getattr(model.config, "num_attention_heads", 32)
        
        layer_entropies = []
        layer_attentions = []
        
        for layer in range(num_layers):
            if layer not in captured_attentions:
                raise RuntimeError(f"Attention weights for layer {layer} were not captured by hooks.")
                
            # shape: (num_heads, num_vision_tokens)
            sliced_attn = captured_attentions[layer][0].float().numpy()
            
            head_entropies = []
            head_attns = []
            
            for head in range(num_heads):
                query_to_vision = sliced_attn[head]
                
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
            
        ret = {
            "num_layers": num_layers,
            "num_heads": num_heads,
            "T": int(T),
            "layer_entropies": layer_entropies, # shape: (layers, heads)
            "layer_attentions": layer_attentions, # shape: (layers, heads, T)
            "max_possible_entropy": float(np.log(T))
        }
        return ret
        
    finally:
        if outputs is not None:
            del outputs
        captured_attentions.clear()
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


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
    parser.add_argument("--prompt-mode", type=str, default="both", choices=["cot", "direct", "both"], help="Prompting mode (cot, direct, or both)")
    args = parser.parse_args()
    
    # Load model and processor once.
    # Note: Experiment 1 needs to extract attention maps (output_attentions=True),
    # which is not supported by PyTorch's SDPA backend. We force "eager" attention.
    model, processor = load_model_and_processor(
        args.model_id, 
        device=args.device, 
        attn_implementation="eager"
    )
    
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
            
            all_results = []
            
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
                inputs = prepare_video_inputs(video_path, question_text, processor, device=args.device, fps=1.0)
                
                try:
                    results = extract_temporal_attention(model, inputs, processor, query_token_pos=-1)
                    results["metadata"] = metadata
                    results["video_name"] = os.path.basename(video_path)
                    results["cohort"] = cohort_label
                    results["prompt"] = question_text
                    
                    # Run text generation to check what the model actually outputs
                    print("    Generating model's reasoning and count answer...")
                    with torch.no_grad():
                        generated_ids = model.generate(**inputs, max_new_tokens=512)
                        generated_ids = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)]
                        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                    results["generated_response"] = generated_text
                    print(f"    Generated Response: {generated_text.strip().replace(chr(10), ' ')}")
                    
                    # Save visual plots for each video with cohort label in filename
                    out_img = os.path.join(output_dir, f"{cohort_label}_{os.path.splitext(os.path.basename(video_path))[0]}_attn.png")
                    plot_attention_dispersion(results, out_img)
                    
                    all_results.append(results)
                except Exception as e:
                    print(f"    Error processing {video_path}: {e}")
                    del e
                finally:
                    # Aggressive cleanup of input and generation tensors
                    if "inputs" in locals() and inputs is not None:
                        for k in list(inputs.keys()):
                            inputs[k] = None
                        inputs = None
                    if "results" in locals():
                        results = None
                    if "generated_ids" in locals():
                        generated_ids = None
                    import gc
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
            if not all_results:
                continue
                
            # Save results summary to JSON
            out_json = os.path.join(output_dir, "attention_dispersion_summary.json")
            summary_results = []
            for r in all_results:
                summary_results.append({
                    "video_name": r["video_name"],
                    "cohort": r.get("cohort", "single_video"),
                    "metadata": r["metadata"],
                    "prompt": r.get("prompt", ""),
                    "generated_response": r.get("generated_response", ""),
                    "layer_entropies": r["layer_entropies"],
                    "max_possible_entropy": r["max_possible_entropy"]
                })
                
            with open(out_json, "w") as f:
                json.dump(summary_results, f, indent=2)
            print(f"Finished attention dispersion analysis for {domain_name} ({prompt_mode})! Summary saved to {out_json}")

if __name__ == "__main__":
    main()
