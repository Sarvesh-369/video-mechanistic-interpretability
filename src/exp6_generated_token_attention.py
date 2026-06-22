import os
import argparse
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from src.utils.model_helpers import load_model_and_processor, prepare_video_inputs, get_associated_files, find_video_files, format_prompt_by_mode

class HookState:
    def __init__(self):
        self.query_pos = -1
        self.captured = {}

def get_hook_fn(layer_idx, state, start_idx, end_idx):
    def hook_fn(module, input_args, output):
        if isinstance(output, tuple) and len(output) > 1 and output[1] is not None:
            attn_weights = output[1] # Shape: (batch_size, num_heads, query_len, seq_len)
            # Slice attention from current query position back to visual token indices
            sliced = attn_weights[:, :, state.query_pos, start_idx:end_idx].detach().clone().cpu()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            state.captured[layer_idx] = sliced
            
            # In-place release to prevent OOM
            dummy = torch.zeros(1, 1, 1, 1, device=attn_weights.device, dtype=attn_weights.dtype)
            attn_weights.set_(dummy)
            new_output = list(output)
            new_output[1] = dummy
            return tuple(new_output)
        return output
    return hook_fn

def load_video_frames(video_path, num_target_frames):
    """
    Reads exactly num_target_frames from the video at uniform intervals
    using decord, torchvision, or opencv to serve as background images.
    """
    # 1. Try decord
    try:
        import decord
        decord.bridge.set_bridge('native')
        vr = decord.VideoReader(video_path)
        total_frames = len(vr)
        indices = np.linspace(0, total_frames - 1, num_target_frames, dtype=int)
        frames = [vr[idx].asnumpy() for idx in indices]
        return frames
    except ImportError:
        pass

    # 2. Try torchvision
    try:
        import torchvision.io as tv_io
        video, _, _ = tv_io.read_video(video_path, pts_unit='sec', output_format='TCHW')
        total_frames = video.shape[0]
        indices = np.linspace(0, total_frames - 1, num_target_frames, dtype=int)
        frames = [video[idx].permute(1, 2, 0).numpy() for idx in indices]
        return frames
    except Exception:
        pass

    # 3. Try cv2 (OpenCV)
    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = set(np.linspace(0, total_frames - 1, num_target_frames, dtype=int))
        frames_dict = {}
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx in indices:
                frames_dict[idx] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            idx += 1
        cap.release()
        frames = [frames_dict[k] for k in sorted(frames_dict.keys())]
        return frames
    except Exception:
        pass

    # 4. Fallback: return dummy frames
    print("Warning: Could not read video frames using decord, torchvision, or cv2. Using dummy frames.")
    return [np.ones((224, 224, 3), dtype=np.uint8) * 255 for _ in range(num_target_frames)]

def run_generated_token_attention(model, inputs, processor, max_new_tokens=60):
    """
    Runs custom autoregressive generation with KV caching and hooks
    to extract visual attention heatmaps (both spatial and temporal)
    for each generated token.
    """
    device = next(model.parameters()).device
    tokenizer = processor.tokenizer
    
    input_ids = inputs["input_ids"][0]
    
    # Locate visual tokens
    vocab = tokenizer.get_vocab()
    vision_start_id = vocab.get("<|vision_start|>", 151652)
    vision_end_id = vocab.get("<|vision_end|>", 151653)
    
    start_pos = (input_ids == vision_start_id).nonzero(as_tuple=True)[0]
    end_pos = (input_ids == vision_end_id).nonzero(as_tuple=True)[0]
    
    if len(start_pos) == 0 or len(end_pos) == 0:
        raise ValueError("Could not locate <|vision_start|> or <|vision_end|> tokens.")
        
    start_idx = start_pos[0].item() + 1
    end_idx = end_pos[0].item()
    
    # Get grid information
    if "video_grid_thw" not in inputs:
        raise ValueError("video_grid_thw not found in inputs.")
    video_grid_thw = inputs["video_grid_thw"][0].cpu().numpy()
    T, H, W = int(video_grid_thw[0]), int(video_grid_thw[1]), int(video_grid_thw[2])
    
    T_out = max(1, T // 2)
    H_out = H // 2
    W_out = W // 2
    expected_tokens = T_out * H_out * W_out
    
    # Locate attention modules
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
        idx = 0
        for name, module in model.named_modules():
            if name.endswith(".self_attn"):
                attn_modules.append((idx, module))
                idx += 1
                
    num_layers = len(attn_modules)
    num_heads = getattr(model.config, "num_attention_heads", 32)
    if hasattr(model.config, "text_config"):
        num_heads = getattr(model.config.text_config, "num_attention_heads", 32)
        
    # Set up hook state
    state = HookState()
    hooks = []
    for layer_idx, module in attn_modules:
        hook = module.register_forward_hook(get_hook_fn(layer_idx, state, start_idx, end_idx))
        hooks.append(hook)
        
    all_step_spatial_attentions = [] # Will store list of shape: (steps, layers, T_out, H_out, W_out)
    generated_token_strs = []
    
    try:
        # Step 0: Prefill (First forward pass)
        print("    Running prefill pass...")
        state.query_pos = -1 # Query of the last token in prompt
        state.captured.clear()
        
        with torch.no_grad():
            outputs = model(**inputs, use_cache=True)
            
        past_key_values = outputs.past_key_values
        next_token_id = torch.argmax(outputs.logits[:, -1, :], dim=-1)
        generated_token_strs.append("<prefill>")
        
        # Pool spatial and temporal prefill attention
        prefill_layers = []
        for layer in range(num_layers):
            sliced = state.captured[layer][0].float().numpy() # shape: (num_heads, V)
            if sliced.shape[1] == expected_tokens:
                reshaped = sliced.reshape(num_heads, T_out, H_out, W_out)
                mean_heads = np.mean(reshaped, axis=0) # shape: (T_out, H_out, W_out)
                prefill_layers.append(mean_heads)
            else:
                prefill_layers.append(np.ones((T_out, H_out, W_out)) / expected_tokens)
        all_step_spatial_attentions.append(prefill_layers)
        
        # Step 1+: Autoregressive generation steps
        stop_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        
        for step in range(max_new_tokens):
            if next_token_id.item() == stop_token_id:
                break
                
            token_str = tokenizer.decode([next_token_id.item()])
            generated_token_strs.append(token_str)
            
            # Reset state for current single-token forward pass
            state.query_pos = 0 # Length of new input_ids is 1
            state.captured.clear()
            
            single_inputs = {
                "input_ids": next_token_id.unsqueeze(0).to(device),
                "past_key_values": past_key_values,
                "use_cache": True
            }
            
            with torch.no_grad():
                outputs = model(**single_inputs)
                
            past_key_values = outputs.past_key_values
            next_token_id = torch.argmax(outputs.logits[:, -1, :], dim=-1)
            
            # Pool spatial and temporal step attention
            step_layers = []
            for layer in range(num_layers):
                if layer not in state.captured:
                    step_layers.append(np.ones((T_out, H_out, W_out)) / expected_tokens)
                    continue
                sliced = state.captured[layer][0].float().numpy() # shape: (num_heads, V)
                if sliced.shape[1] == expected_tokens:
                    reshaped = sliced.reshape(num_heads, T_out, H_out, W_out)
                    mean_heads = np.mean(reshaped, axis=0) # shape: (T_out, H_out, W_out)
                    step_layers.append(mean_heads)
                else:
                    step_layers.append(np.ones((T_out, H_out, W_out)) / expected_tokens)
            all_step_spatial_attentions.append(step_layers)
            
            # Print token generated in progress
            print(f"      [Step {step+1}] Generated Token: {repr(token_str)}")
            
    finally:
        for hook in hooks:
            hook.remove()
            
    # Cleanup memory
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    return {
        "generated_tokens": generated_token_strs,
        "attentions": all_step_spatial_attentions, # shape: (steps, layers, T_out, H_out, W_out)
        "num_layers": num_layers,
        "num_heads": num_heads,
        "T": T,
        "T_out": T_out,
        "H_out": H_out,
        "W_out": W_out
    }

def plot_generated_token_temporal_heatmap(results, output_image_path, target_layer=-2, duration=24.0):
    """
    Generates a 2D heatmap showing visual attention over video steps (x-axis)
    across each generated token (y-axis) for a specified transformer layer.
    Plots exactly the T_out steps given to the decoder (no upscaling).
    """
    tokens = results["generated_tokens"]
    attentions = np.array(results["attentions"]) # shape: (steps, layers, T_out, H_out, W_out)
    T_out = results["T_out"]
    
    num_layers = results["num_layers"]
    layer_idx = target_layer if target_layer >= 0 else num_layers + target_layer
    
    # Sum over spatial coordinates (H_out, W_out) to get temporal attention
    temporal_layer_attn = np.sum(attentions[:, layer_idx, :, :, :], axis=(3, 4)) # shape: (steps, T_out)
    
    # Normalize over T_out for each step
    normalized_attn = []
    for step in range(len(tokens)):
        step_val = temporal_layer_attn[step]
        sum_val = np.sum(step_val)
        norm_val = step_val / sum_val if sum_val > 0 else np.ones(T_out) / T_out
        normalized_attn.append(norm_val)
    normalized_attn = np.array(normalized_attn) # shape: (steps, T_out)
    
    fig, ax = plt.subplots(figsize=(12, len(tokens) * 0.25 + 2))
    im = ax.imshow(normalized_attn, aspect='auto', cmap='viridis', origin='upper')
    
    ax.set_yticks(range(len(tokens)))
    ax.set_yticklabels(tokens, fontsize=8)
    
    # Generate timestamp labels for the x-axis mapping each step to seconds
    x_ticks = range(T_out)
    x_labels = [f"{t * (duration / T_out):.1f}s" for t in x_ticks]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, fontsize=8, rotation=45)
    
    ax.set_xlabel("Decoder Visual Steps (Seconds)", fontsize=10, fontweight='bold')
    ax.set_ylabel("Generated Tokens (Autoregressive Sequence)", fontsize=10, fontweight='bold')
    ax.set_title(f"Dynamic Visual Attention per Generated Token (Layer {layer_idx}, T_out={T_out} steps)", fontsize=12, fontweight='bold')
    
    fig.colorbar(im, ax=ax, label="Attention Weight")
    plt.tight_layout()
    plt.savefig(output_image_path, dpi=300)
    plt.close()

def plot_spatial_attention_for_token(results, token_idx, video_path, output_image_path, target_layer=-2, duration=24.0):
    """
    Plots the spatial attention visual heatmaps across all video steps
    for a specific generated token, overlaid on the original video frames.
    Plots exactly the T_out steps (no upscaling).
    """
    tokens = results["generated_tokens"]
    if token_idx < 0 or token_idx >= len(tokens):
        print(f"Error: Token index {token_idx} out of range (total generated: {len(tokens)})")
        return
        
    token_str = tokens[token_idx]
    attentions = results["attentions"][token_idx] # shape: (layers, T_out, H_out, W_out)
    T_out = results["T_out"]
    
    num_layers = results["num_layers"]
    layer_idx = target_layer if target_layer >= 0 else num_layers + target_layer
    
    # shape: (T_out, H_out, W_out)
    token_layer_attn = attentions[layer_idx]
    
    # Load exactly T_out frames from the video to serve as background
    print(f"    Loading {T_out} video frames for heatmap overlay...")
    background_frames = load_video_frames(video_path, T_out)
    
    # Select a subset of frames to display (max 16 subplots for clean rendering)
    max_subplots = 16
    step_size = max(1, T_out // max_subplots)
    frame_indices = list(range(0, T_out, step_size))[:max_subplots]
    
    cols = min(4, len(frame_indices))
    rows = int(np.ceil(len(frame_indices) / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))
    axes = axes.flatten() if len(frame_indices) > 1 else [axes]
    
    # Determine global min/max for visual comparison
    vmin = np.min(token_layer_attn)
    vmax = np.max(token_layer_attn)
    if vmax == vmin:
        vmax += 1e-9
        
    for idx, t in enumerate(frame_indices):
        ax = axes[idx]
        spatial_map = token_layer_attn[t] # shape: (H_out, W_out)
        background = background_frames[t]
        timestamp = t * (duration / T_out)
        
        # Plot original frame as background
        ax.imshow(background)
        # Overlay attention heatmap with bilinear interpolation and transparency (alpha=0.5)
        im = ax.imshow(spatial_map, cmap='jet', alpha=0.5, 
                      extent=[0, background.shape[1], background.shape[0], 0], 
                      interpolation='bilinear', vmin=vmin, vmax=vmax)
        
        ax.set_title(f"Time: {timestamp:.1f}s (Step {t})", fontsize=9)
        ax.axis('off')
        
    # Turn off unused subplots
    for idx in range(len(frame_indices), len(axes)):
        axes[idx].axis('off')
        
    fig.suptitle(f"Spatial Visual Attention for Token [{token_idx}] '{token_str}' (Layer {layer_idx})", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_image_path, dpi=300)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Run Experiment 6: Generated Token Visual Attention Tracking")
    parser.add_argument("--video-path", type=str, default=None, help="Path to video file")
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen3-VL-8B-Instruct", help="HF model ID")
    parser.add_argument("--output-dir", type=str, default="results/exp6", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda", help="Target device")
    parser.add_argument("--prompt-mode", type=str, default="cot", choices=["cot", "direct"], help="Prompting mode")
    parser.add_argument("--max-new-tokens", type=int, default=60, help="Max tokens to generate")
    args = parser.parse_args()
    
    model, processor = load_model_and_processor(
        args.model_id,
        device=args.device,
        attn_implementation="eager"
    )
    
    if args.video_path:
        instances = [get_associated_files(args.video_path)]
    else:
        # Default to a representative video with f <= 2.0 and c <= 4 (easy/medium regime)
        default_dir = "videos/temporal/blinking"
        if os.path.exists(default_dir):
            all_instances = find_video_files(default_dir)
            easy_instances = [
                inst for inst in all_instances 
                if inst["metadata"]["count"] is not None 
                and inst["metadata"]["count"] <= 4
                and inst["metadata"]["frequency"] is not None
                and inst["metadata"]["frequency"] <= 2.0
            ]
            if easy_instances:
                instances = easy_instances[:1]
            else:
                instances = all_instances[:1]
        else:
            print("Error: No video specified and default video directories not found.")
            return
            
    if not instances:
        print("Error: No instances resolved for running experiment.")
        return
        
    os.makedirs(args.output_dir, exist_ok=True)
    
    for inst in instances:
        video_path = inst["video_path"]
        q_path = inst["question_path"]
        metadata = inst["metadata"]
        duration = metadata.get("duration", 24.0)
        
        if not q_path:
            raw_question = "How many times did the object flash?"
        else:
            with open(q_path, "r") as f:
                raw_question = f.read().strip()
                
        question_text = format_prompt_by_mode(raw_question, args.prompt_mode)
        # Prepare visual inputs at 2.0 FPS
        inputs = prepare_video_inputs(video_path, question_text, processor, device=args.device, fps=2.0)
        
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        print(f"\nProcessing Generated Token Attention for {video_name} (Mode: {args.prompt_mode.upper()}, 2.0 FPS)...")
        
        try:
            results = run_generated_token_attention(model, inputs, processor, max_new_tokens=args.max_new_tokens)
            results["video_name"] = os.path.basename(video_path)
            results["prompt"] = question_text
            
            # Save temporal attention shift plots (no upscaling)
            for target_layer in [-1, -2]:
                out_img = os.path.join(args.output_dir, f"{video_name}_{args.prompt_mode}_layer{target_layer}_temporal_attention_shift.png")
                plot_generated_token_temporal_heatmap(results, out_img, target_layer=target_layer, duration=duration)
                print(f"    Saved temporal shift plot for Layer {target_layer} to {out_img}")
                
            # Plot spatial visual heatmaps overlaid on actual video frames
            tokens = results["generated_tokens"]
            print(f"    Generated {len(tokens)} tokens in total.")
            
            target_indices = [idx for idx in [5, 10, 15, 20, 30] if idx < len(tokens)]
            for token_idx in target_indices:
                safe_token_name = "".join([c if c.isalnum() else "_" for c in tokens[token_idx]]).strip("_")
                if not safe_token_name:
                    safe_token_name = f"idx{token_idx}"
                out_img = os.path.join(args.output_dir, f"{video_name}_{args.prompt_mode}_token{token_idx}_{safe_token_name}_spatial_heatmap.png")
                plot_spatial_attention_for_token(results, token_idx, video_path, out_img, target_layer=-2, duration=duration)
                print(f"    Saved spatial visual heatmap overlay for token [{token_idx}] '{tokens[token_idx]}' to {out_img}")
                
            # Save results dictionary to JSON
            summary_results = {
                "video_name": results["video_name"],
                "prompt": results["prompt"],
                "generated_tokens": results["generated_tokens"],
                "T": results["T"],
                "T_out": results["T_out"],
                "H_out": results["H_out"],
                "W_out": results["W_out"],
                "num_layers": results["num_layers"],
                "num_heads": results["num_heads"]
            }
            out_json = os.path.join(args.output_dir, f"{video_name}_{args.prompt_mode}_attention_shift.json")
            with open(out_json, "w") as f:
                json.dump(summary_results, f, indent=2)
            print(f"    Saved summary metadata to {out_json}")
            
        except Exception as e:
            print(f"    Error processing video {video_name}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
