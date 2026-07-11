import os
import argparse
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from src.utils.model_helpers import load_model_and_processor, prepare_video_inputs, get_associated_files, find_video_files, format_prompt_by_mode

class AttnCaptureState:
    def __init__(self):
        self.enabled = False
        self.temp_dir = None
        self.captured_files = {} # layer_idx -> file_path
        self.captured_tensors = {} # layer_idx -> tensor (shape: (q_len, seq_len))
        self.decode_attns = {} # layer_idx -> list of tensors

attn_capture_state = AttnCaptureState()

def compute_rollout_for_position(target_position, target_layer, num_layers, prompt_len, decode_attns, captured_files, visual_positions, T_out, H_out, W_out):
    """
    Computes target-specific attention rollout for a target sequence position
    at a target layer index (0 to num_layers-1), mapping back to the visual tokens.
    """
    curr_len = target_position + 1
    relevance = torch.zeros(curr_len, dtype=torch.float32)
    relevance[target_position] = 1.0
    
    # Move backward from target_layer down to 0
    for l in range(target_layer, -1, -1):
        # Construct the attention matrix A of shape (curr_len, curr_len)
        A = torch.zeros((curr_len, curr_len), dtype=torch.float32)
        
        # 1. Load prefill submatrix
        if l in captured_files and os.path.exists(captured_files[l]):
            try:
                prefill_attn = torch.load(captured_files[l], map_location="cpu").float()
                p_len = min(prompt_len, curr_len)
                A[:p_len, :p_len] = prefill_attn[:p_len, :p_len]
            except Exception as e:
                pass
                
        # 2. Fill decode rows
        if l in decode_attns:
            for k, row in enumerate(decode_attns[l]):
                row_idx = prompt_len + k
                if row_idx >= curr_len:
                    break
                row_val = row[0].float() # (seq_len,)
                r_len = min(row_val.shape[0], curr_len)
                A[row_idx, :r_len] = row_val[:r_len]
                
        # 3. Add identity matrix for residual path
        identity = torch.eye(curr_len, dtype=torch.float32)
        A = A + identity
        
        # 4. Row normalize
        row_sums = A.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        A = A / row_sums
        
        # 5. Multiply relevance vector
        relevance = relevance @ A
        
    # Extract only the visual token positions
    valid_vis_pos = [pos for pos in visual_positions if pos < curr_len]
    if not valid_vis_pos:
        return np.zeros((T_out, H_out, W_out), dtype=np.float32)
        
    vis_relevance = relevance[valid_vis_pos].numpy()
    
    # Pad if some visual positions were not reached yet due to causality
    if len(vis_relevance) < len(visual_positions):
        padded = np.zeros(len(visual_positions), dtype=np.float32)
        padded[:len(vis_relevance)] = vis_relevance
        vis_relevance = padded
        
    # Fallback/Safety Check: Ensure dimensions match exactly T_out * H_out * W_out
    expected_size = T_out * H_out * W_out
    if len(vis_relevance) != expected_size:
        if len(vis_relevance) > expected_size:
            vis_relevance = vis_relevance[:expected_size]
        else:
            padded = np.zeros(expected_size, dtype=np.float32)
            padded[:len(vis_relevance)] = vis_relevance
            vis_relevance = padded
            
    # Reshape to (T_out, H_out, W_out)
    return vis_relevance.reshape(T_out, H_out, W_out)

def compute_raw_attention_for_position(target_position, target_layer, prompt_len, decode_attns, captured_files, visual_positions, T_out, H_out, W_out):
    """
    Computes raw attention mapping back to the visual tokens at a specific target layer and sequence position.
    """
    V = len(visual_positions)
    if target_position < prompt_len:
        # Prefill stage: read from saved file
        if target_layer in captured_files and os.path.exists(captured_files[target_layer]):
            try:
                prefill_attn = torch.load(captured_files[target_layer], map_location="cpu").float()
                # Last prompt token is prompt_len - 1
                attn_vector = prefill_attn[-1, visual_positions].numpy()
            except Exception:
                attn_vector = np.ones(V, dtype=np.float32) / V
        else:
            attn_vector = np.ones(V, dtype=np.float32) / V
    else:
        # Decode stage: read from buffer
        step_idx = target_position - prompt_len
        if target_layer in decode_attns and step_idx < len(decode_attns[target_layer]):
            try:
                row_val = decode_attns[target_layer][step_idx][0].float() # (seq_len,)
                attn_vector = row_val[visual_positions].numpy()
            except Exception:
                attn_vector = np.ones(V, dtype=np.float32) / V
        else:
            attn_vector = np.ones(V, dtype=np.float32) / V
            
    # Fallback/Safety Check: Ensure dimensions match exactly T_out * H_out * W_out
    expected_size = T_out * H_out * W_out
    if len(attn_vector) != expected_size:
        if len(attn_vector) > expected_size:
            attn_vector = attn_vector[:expected_size]
        else:
            padded = np.zeros(expected_size, dtype=np.float32)
            padded[:len(attn_vector)] = attn_vector
            attn_vector = padded
            
    return attn_vector.reshape(T_out, H_out, W_out)

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, seqlen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, seqlen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, seqlen, head_dim)

def custom_eager_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor,
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    bs, num_heads, q_len, head_dim = query.shape
    seq_len = key_states.shape[-2]
    
    chunk_size = 512 # small enough to avoid OOM
    
    if q_len > chunk_size:
        # Prefill / large sequence: run query chunking
        attn_outputs = []
        mean_heads_list = []
        
        for start_idx in range(0, q_len, chunk_size):
            end_idx = min(start_idx + chunk_size, q_len)
            query_chunk = query[:, :, start_idx:end_idx, :]
            
            # Compute raw attention scores for the chunk
            attn_weights_chunk = torch.matmul(query_chunk, key_states.transpose(2, 3)) * scaling
            
            if attention_mask is not None:
                if attention_mask.dim() == 4:
                    causal_mask_chunk = attention_mask[:, :, start_idx:end_idx, :seq_len]
                elif attention_mask.dim() == 3:
                    causal_mask_chunk = attention_mask[:, start_idx:end_idx, :seq_len]
                else:
                    causal_mask_chunk = attention_mask[start_idx:end_idx, :seq_len]
                attn_weights_chunk = attn_weights_chunk + causal_mask_chunk
                
            # Softmax in float32 for numerical stability
            attn_weights_chunk = torch.nn.functional.softmax(attn_weights_chunk, dim=-1, dtype=torch.float32).to(query.dtype)
            
            if dropout > 0.0:
                attn_weights_chunk = torch.nn.functional.dropout(attn_weights_chunk, p=dropout, training=module.training)
                
            # Compute output chunk
            attn_output_chunk = torch.matmul(attn_weights_chunk, value_states)
            attn_outputs.append(attn_output_chunk)
            
            # If capture is enabled, save the mean heads for rollout
            if attn_capture_state.enabled:
                mean_heads_chunk = torch.mean(attn_weights_chunk, dim=1).detach().cpu()
                if bs == 1:
                    mean_heads_chunk = mean_heads_chunk[0] # (chunk_len, seq_len)
                mean_heads_list.append(mean_heads_chunk)
                
            # Explicitly delete chunk tensors to free GPU memory
            del attn_weights_chunk
            del query_chunk
            
        attn_output = torch.cat(attn_outputs, dim=2)
        attn_output = attn_output.transpose(1, 2).contiguous()
        
        # Save prefill attention weights to disk
        if attn_capture_state.enabled and len(mean_heads_list) > 0:
            layer_idx = getattr(module, "layer_idx", None)
            if layer_idx is not None:
                mean_heads_all = torch.cat(mean_heads_list, dim=0) # (q_len, seq_len)
                file_path = os.path.join(attn_capture_state.temp_dir, f"prefill_attn_layer_{layer_idx}.pt")
                torch.save(mean_heads_all.half(), file_path)
                attn_capture_state.captured_files[layer_idx] = file_path
                
    else:
        # Small query length (e.g. decoding steps, q_len=1)
        attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
        if attention_mask is not None:
            if attention_mask.dim() == 4:
                causal_mask = attention_mask[:, :, :, :seq_len]
            elif attention_mask.dim() == 3:
                causal_mask = attention_mask[:, :, :seq_len]
            else:
                causal_mask = attention_mask[:, :seq_len]
            attn_weights = attn_weights + causal_mask
            
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        if dropout > 0.0:
            attn_weights = torch.nn.functional.dropout(attn_weights, p=dropout, training=module.training)
            
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        
        if attn_capture_state.enabled:
            layer_idx = getattr(module, "layer_idx", None)
            if layer_idx is not None:
                mean_heads = torch.mean(attn_weights, dim=1).detach().cpu()
                if bs == 1:
                    mean_heads = mean_heads[0] # (q_len, seq_len)
                attn_capture_state.captured_tensors[layer_idx] = mean_heads.half()
                
        del attn_weights

    dummy = torch.zeros(1, 1, 1, 1, device=query.device, dtype=query.dtype)
    return attn_output, dummy

# Monkeypatching eager attention forward for Qwen3-VL, Qwen2.5-VL, or Qwen2-VL
patched = False
try:
    import transformers.models.qwen3_vl.modeling_qwen3_vl as modeling_qwen3_vl
    modeling_qwen3_vl.eager_attention_forward = custom_eager_attention_forward
    print("Successfully monkeypatched Qwen3-VL eager_attention_forward.")
    patched = True
except ImportError:
    pass

try:
    import transformers.models.qwen2_5_vl.modeling_qwen2_5_vl as modeling_qwen2_5_vl
    modeling_qwen2_5_vl.eager_attention_forward = custom_eager_attention_forward
    print("Successfully monkeypatched Qwen2.5-VL eager_attention_forward.")
    patched = True
except ImportError:
    pass

try:
    import transformers.models.qwen2_vl.modeling_qwen2_vl as modeling_qwen2_vl
    modeling_qwen2_vl.eager_attention_forward = custom_eager_attention_forward
    print("Successfully monkeypatched Qwen2-VL eager_attention_forward.")
    patched = True
except ImportError:
    pass

if not patched:
    print("Warning: Could not import or patch any Qwen-VL eager_attention_forward modules.")

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

def run_generated_token_attention_rollout(model, inputs, processor, max_new_tokens=60, forced_token_ids=None):
    """
    Runs custom autoregressive generation with KV caching and offloaded
    attention weights to compute the formal Causal Attention Rollout mapping
    back to the input visual tokens for each generated token.
    Supports teacher-forced decoding if forced_token_ids is provided.
    """
    device = next(model.parameters()).device
    tokenizer = processor.tokenizer
    
    input_ids = inputs["input_ids"][0]
    prompt_len = input_ids.shape[0]
    
    # Locate visual tokens
    vocab = tokenizer.get_vocab()
    vision_start_id = vocab.get("<|vision_start|>", 151652)
    vision_end_id = vocab.get("<|vision_end|>", 151653)
    video_pad_id = vocab.get("<|video_pad|>", None)
    image_pad_id = vocab.get("<|image_pad|>", None)
    if video_pad_id is None:
        video_pad_id = getattr(processor, "video_token_id", None)
    if image_pad_id is None:
        image_pad_id = getattr(processor, "image_token_id", None)
        
    visual_ids = [vid for vid in [video_pad_id, image_pad_id] if vid is not None]
    
    start_pos = (input_ids == vision_start_id).nonzero(as_tuple=True)[0]
    end_pos = (input_ids == vision_end_id).nonzero(as_tuple=True)[0]
    
    if len(start_pos) == 0 or len(end_pos) == 0:
        raise ValueError("Could not locate <|vision_start|> or <|vision_end|> tokens.")
        
    visual_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for vid in visual_ids:
        visual_mask = visual_mask | (input_ids == vid)
        
    in_vision_bounds = torch.zeros_like(input_ids, dtype=torch.bool)
    for s_pos, e_pos in zip(start_pos, end_pos):
        in_vision_bounds[s_pos.item() + 1 : e_pos.item()] = True
    visual_mask = visual_mask & in_vision_bounds
    
    visual_positions = visual_mask.nonzero(as_tuple=True)[0].cpu().numpy().tolist()
    V = len(visual_positions)
    
    # Get grid information
    if "video_grid_thw" not in inputs:
        raise ValueError("video_grid_thw not found in inputs.")
    video_grid_thw = inputs["video_grid_thw"][0].cpu().numpy()
    T, H, W = int(video_grid_thw[0]), int(video_grid_thw[1]), int(video_grid_thw[2])
    
    T_out = max(1, T // 2)
    H_out = H // 2
    W_out = W // 2
    
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
    
    # Ensure layer_idx is set on all attention modules
    for idx, module in attn_modules:
        if not hasattr(module, "layer_idx") or module.layer_idx is None:
            module.layer_idx = idx
            
    # Raw attention tracking variables
    all_step_spatial_attns = [] # Will store raw visual attention maps: (steps, layers, T, H_out, W_out)
    generated_token_strs = []
    
    try:
        # Enable attention capture
        attn_capture_state.enabled = True
        attn_capture_state.captured_files.clear()
        attn_capture_state.captured_tensors.clear()
        attn_capture_state.decode_attns = {layer_idx: [] for layer_idx in range(num_layers)}
        
        # Step 0: Prefill (First forward pass)
        print("    Running prefill pass...")
        
        with torch.no_grad():
            outputs = model(**inputs, use_cache=True)
            
        past_key_values = outputs.past_key_values
        if forced_token_ids is not None and len(forced_token_ids) > 0:
            next_token_id = torch.tensor([forced_token_ids[0]], device=device)
        else:
            next_token_id = torch.argmax(outputs.logits[:, -1, :], dim=-1)
        generated_token_strs.append("<prefill>")
        
        # Step 1+: Autoregressive generation steps
        stop_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        
        loop_limit = len(forced_token_ids) if forced_token_ids is not None else max_new_tokens
        
        for step in range(loop_limit):
            if forced_token_ids is None and next_token_id.item() == stop_token_id:
                break
                
            token_str = tokenizer.decode([next_token_id.item()])
            generated_token_strs.append(token_str)
            
            # Reset state for current single-token forward pass
            attn_capture_state.captured_tensors.clear()
            
            single_inputs = {
                "input_ids": next_token_id.unsqueeze(0).to(device),
                "past_key_values": past_key_values,
                "use_cache": True
            }
            
            with torch.no_grad():
                outputs = model(**single_inputs)
                
            past_key_values = outputs.past_key_values
            
            # Save the captured decode attention rows for rollout
            for layer_idx in range(num_layers):
                if layer_idx in attn_capture_state.captured_tensors:
                    attn_row = attn_capture_state.captured_tensors[layer_idx] # shape: (1, seq_len)
                    attn_capture_state.decode_attns[layer_idx].append(attn_row.clone())
            
            # Print token generated in progress
            print(f"      [Step {step+1}/{loop_limit}] Generated Token: {repr(token_str)}")
            
            if forced_token_ids is not None:
                if step < len(forced_token_ids) - 1:
                    next_token_id = torch.tensor([forced_token_ids[step + 1]], device=device)
                else:
                    break
            else:
                next_token_id = torch.argmax(outputs.logits[:, -1, :], dim=-1)
                
        # Generation complete. Now compute attention rollout and raw attention for all steps and layers
        print("    Computing layer-wise attention rollout and raw attention...")
        all_step_raw_spatial_attns = []
        for step_idx in range(len(generated_token_strs)):
            target_pos = prompt_len - 1 if step_idx <= 1 else prompt_len - 2 + step_idx
            
            step_rollouts = []
            step_raws = []
            for layer_idx in range(num_layers):
                rollout_map = compute_rollout_for_position(
                    target_position=target_pos,
                    target_layer=layer_idx,
                    num_layers=num_layers,
                    prompt_len=prompt_len,
                    decode_attns=attn_capture_state.decode_attns,
                    captured_files=attn_capture_state.captured_files,
                    visual_positions=visual_positions,
                    T_out=T_out,
                    H_out=H_out,
                    W_out=W_out
                )
                raw_map = compute_raw_attention_for_position(
                    target_position=target_pos,
                    target_layer=layer_idx,
                    prompt_len=prompt_len,
                    decode_attns=attn_capture_state.decode_attns,
                    captured_files=attn_capture_state.captured_files,
                    visual_positions=visual_positions,
                    T_out=T_out,
                    H_out=H_out,
                    W_out=W_out
                )
                step_rollouts.append(rollout_map)
                step_raws.append(raw_map)
            all_step_spatial_attns.append(step_rollouts)
            all_step_raw_spatial_attns.append(step_raws)
            
    finally:
        # Disable attention capture
        attn_capture_state.enabled = False
        attn_capture_state.captured_files.clear()
        attn_capture_state.captured_tensors.clear()
        if hasattr(attn_capture_state, "decode_attns"):
            attn_capture_state.decode_attns.clear()
        
        # Clean up temp directory
        if attn_capture_state.temp_dir and os.path.exists(attn_capture_state.temp_dir):
            import shutil
            try:
                shutil.rmtree(attn_capture_state.temp_dir)
            except Exception as e:
                pass
            
    # Cleanup memory
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    return {
        "generated_tokens": generated_token_strs,
        "attentions": all_step_spatial_attns, # shape: (steps, layers, T_out, H_out, W_out)
        "raw_attentions": all_step_raw_spatial_attns, # shape: (steps, layers, T_out, H_out, W_out)
        "num_layers": num_layers,
        "T": T,
        "T_out": T_out,
        "H_out": H_out,
        "W_out": W_out
    }

def plot_generated_token_temporal_heatmap(results, output_image_path, target_layer=-2, duration=24.0):
    """
    Generates a 2D heatmap showing visual attention rollout over video steps (x-axis)
    across each generated token (y-axis) for a specified transformer layer.
    Plots exactly the T_out steps given to the decoder (no upscaling).
    """
    tokens = results["generated_tokens"]
    attentions = np.array(results["attentions"]) # shape: (steps, layers, T, H_out, W_out)
    T_actual = attentions.shape[2]
    
    num_layers = results["num_layers"]
    layer_idx = target_layer if target_layer >= 0 else num_layers + target_layer
    
    # Sum over spatial coordinates (H_out, W_out) to get temporal attention
    temporal_layer_attn = np.sum(attentions[:, layer_idx, :, :, :], axis=(2, 3)) # shape: (steps, T)
    
    # Normalize over temporal dimension for each step
    normalized_attn = []
    for step in range(len(tokens)):
        step_val = temporal_layer_attn[step]
        sum_val = np.sum(step_val)
        norm_val = step_val / sum_val if sum_val > 0 else np.ones(T_actual) / T_actual
        normalized_attn.append(norm_val)
    normalized_attn = np.array(normalized_attn) # shape: (steps, T)
    
    fig, ax = plt.subplots(figsize=(12, len(tokens) * 0.25 + 2))
    im = ax.imshow(normalized_attn, aspect='auto', cmap='viridis', origin='upper')
    
    ax.set_yticks(range(len(tokens)))
    ax.set_yticklabels(tokens, fontsize=8)
    
    # Generate timestamp labels for the x-axis mapping each step to seconds
    x_ticks = range(T_actual)
    x_labels = [f"{t * (duration / T_actual):.1f}s" for t in x_ticks]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, fontsize=8, rotation=45)
    
    ax.set_xlabel("Decoder Visual Steps (Seconds)", fontsize=10, fontweight='bold')
    ax.set_ylabel("Generated Tokens (Autoregressive Sequence)", fontsize=10, fontweight='bold')
    ax.set_title(f"Dynamic Visual Attention Rollout per Generated Token (Layer {layer_idx}, T={T_actual} steps)", fontsize=12, fontweight='bold')
    
    fig.colorbar(im, ax=ax, label="Attention Flow")
    plt.tight_layout()
    plt.savefig(output_image_path, dpi=300)
    plt.close()

def plot_spatial_attention_for_token(results, token_idx, video_path, output_image_path, target_layer=-2, duration=24.0):
    """
    Plots the spatial attention visual rollout heatmaps across all video steps
    for a specific generated token, overlaid on the original video frames.
    Plots exactly the T_out steps (no upscaling).
    """
    tokens = results["generated_tokens"]
    if token_idx < 0 or token_idx >= len(tokens):
        print(f"Error: Token index {token_idx} out of range (total generated: {len(tokens)})")
        return
        
    token_str = tokens[token_idx]
    num_layers = results["num_layers"]
    layer_idx = target_layer if target_layer >= 0 else num_layers + target_layer
    token_layer_attn = np.array(results["attentions"][token_idx][layer_idx])
    T_actual = token_layer_attn.shape[0]
    
    # Load all T_actual frames from the video to serve as background
    print(f"    Loading {T_actual} video frames for heatmap overlay...")
    background_frames = load_video_frames(video_path, T_actual)
    
    cols = 4
    rows = int(np.ceil(T_actual / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.0, rows * 3.0))
    axes = axes.flatten() if T_actual > 1 else [axes]
    
    # Determine global min/max for visual comparison
    vmin = np.min(token_layer_attn)
    vmax = np.max(token_layer_attn)
    if vmax == vmin:
        vmax += 1e-9
        
    for t in range(T_actual):
        ax = axes[t]
        spatial_map = token_layer_attn[t] # shape: (H_out, W_out)
        background = background_frames[t]
        timestamp = t * (duration / T_actual)
        
        # Plot original frame as background
        ax.imshow(background)
        # Overlay attention rollout heatmap with bilinear interpolation and transparency (alpha=0.5)
        im = ax.imshow(spatial_map, cmap='jet', alpha=0.5, 
                      extent=[0, background.shape[1], background.shape[0], 0], 
                      interpolation='bilinear', vmin=vmin, vmax=vmax)
        
        ax.set_title(f"Time: {timestamp:.1f}s (Step {t})", fontsize=9)
        ax.axis('off')
        
    # Turn off unused subplots
    for idx in range(T_actual, len(axes)):
        axes[idx].axis('off')
        
    fig.suptitle(f"Visual Attention Rollout for Token [{token_idx}] '{token_str}' (Layer {layer_idx})", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_image_path, dpi=300)
    plt.close()

def plot_layer_by_frame_heatmap(results, token_idx, output_image_path, duration=24.0):
    """
    Plots a layer-by-frame heatmap (M_{l,t}) showing attention rollout relevance 
    over video steps (x-axis) across all layers (y-axis) for a specific generated token.
    """
    tokens = results["generated_tokens"]
    token_str = tokens[token_idx]
    
    # attentions shape: (steps, layers, T_out, H_out, W_out)
    attentions = np.array(results["attentions"])
    token_attn = attentions[token_idx] # (layers, T_out, H_out, W_out)
    
    # Sum over spatial dimensions (H_out, W_out)
    M = np.sum(token_attn, axis=(2, 3)) # (layers, T_out)
    
    # Normalize per layer
    M_norm = []
    for l in range(M.shape[0]):
        layer_sum = np.sum(M[l])
        if layer_sum > 0:
            M_norm.append(M[l] / layer_sum)
        else:
            M_norm.append(np.ones(M.shape[1]) / M.shape[1])
    M_norm = np.array(M_norm) # (layers, T_out)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(M_norm, aspect='auto', cmap='viridis', origin='lower')
    
    ax.set_xlabel("Video Steps (Seconds)", fontsize=10, fontweight='bold')
    ax.set_ylabel("Transformer Layer Index", fontsize=10, fontweight='bold')
    ax.set_title(f"Layer-by-Frame Attention Rollout for Token [{token_idx}] '{token_str}'", fontsize=12, fontweight='bold')
    
    T_actual = M_norm.shape[1]
    x_ticks = range(T_actual)
    x_labels = [f"{t * (duration / T_actual):.1f}s" for t in x_ticks]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, fontsize=8, rotation=45)
    
    fig.colorbar(im, ax=ax, label="Normalized Relevance")
    plt.tight_layout()
    plt.savefig(output_image_path, dpi=300)
    plt.close()

def plot_frame_level_rollout_curve(results, token_idx, trace_path, output_image_path, target_layer=-2, duration=24.0):
    """
    Plots a line curve (t -> R_t) of attention rollout over time with vertical lines for events.
    """
    tokens = results["generated_tokens"]
    token_str = tokens[token_idx]
    num_layers = results["num_layers"]
    layer_idx = target_layer if target_layer >= 0 else num_layers + target_layer
    
    attentions = np.array(results["attentions"])
    token_layer_attn = attentions[token_idx][layer_idx] # (T_out, H_out, W_out)
    R_t = np.sum(token_layer_attn, axis=(1, 2)) # (T_out,)
    
    # Normalize
    r_sum = np.sum(R_t)
    if r_sum > 0:
        R_t = R_t / r_sum
        
    T_actual = len(R_t)
    times = [t * (duration / T_actual) for t in range(T_actual)]
    
    # Parse event times
    event_times = []
    if trace_path and os.path.exists(trace_path):
        try:
            with open(trace_path, "r") as f:
                trace_data = json.load(f)
            event_times = [e[0] for e in trace_data.get("events", [])]
        except Exception:
            pass
            
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(times, R_t, label="Attention Rollout Relevance", color="blue", linewidth=2)
    
    # Draw vertical lines for event moments
    for et in event_times:
        ax.axvline(x=et, color="red", linestyle="--", alpha=0.8, label="Event Moment" if et == event_times[0] else "")
        
    ax.set_xlabel("Time (Seconds)", fontsize=10, fontweight='bold')
    ax.set_ylabel("Relevance score", fontsize=10, fontweight='bold')
    ax.set_title(f"Frame-level Attention Rollout Curve for Token '{token_str}' (Layer {layer_idx})", fontsize=12, fontweight='bold')
    ax.grid(True, linestyle=":", alpha=0.6)
    if event_times:
        ax.legend()
    plt.tight_layout()
    plt.savefig(output_image_path, dpi=300)
    plt.close()

def plot_raw_vs_rollout_comparison(results, token_idx, trace_path, output_image_path, target_layer=-2, duration=24.0):
    """
    Plots temporal attention curve comparing raw attention vs. rollout on the same plot.
    """
    tokens = results["generated_tokens"]
    token_str = tokens[token_idx]
    num_layers = results["num_layers"]
    layer_idx = target_layer if target_layer >= 0 else num_layers + target_layer
    
    # Rollout
    attentions = np.array(results["attentions"])
    token_layer_attn = attentions[token_idx][layer_idx] # (T_out, H_out, W_out)
    R_t = np.sum(token_layer_attn, axis=(1, 2))
    r_sum = np.sum(R_t)
    if r_sum > 0:
        R_t = R_t / r_sum
        
    # Raw Attention
    raw_attentions = np.array(results["raw_attentions"])
    token_layer_raw = raw_attentions[token_idx][layer_idx] # (T_out, H_out, W_out)
    Raw_t = np.sum(token_layer_raw, axis=(1, 2))
    raw_sum = np.sum(Raw_t)
    if raw_sum > 0:
        Raw_t = Raw_t / raw_sum
        
    T_actual = len(R_t)
    times = [t * (duration / T_actual) for t in range(T_actual)]
    
    # Parse event times
    event_times = []
    if trace_path and os.path.exists(trace_path):
        try:
            with open(trace_path, "r") as f:
                trace_data = json.load(f)
            event_times = [e[0] for e in trace_data.get("events", [])]
        except Exception:
            pass
            
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(times, R_t, label="Causal Rollout", color="blue", linewidth=2)
    ax.plot(times, Raw_t, label="Raw Attention", color="green", linestyle="-.", linewidth=1.5)
    
    for et in event_times:
        ax.axvline(x=et, color="red", linestyle="--", alpha=0.8, label="Event Moment" if et == event_times[0] else "")
        
    ax.set_xlabel("Time (Seconds)", fontsize=10, fontweight='bold')
    ax.set_ylabel("Attention Score", fontsize=10, fontweight='bold')
    ax.set_title(f"Raw Attention vs. Rollout for Token '{token_str}' (Layer {layer_idx})", fontsize=12, fontweight='bold')
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_image_path, dpi=300)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Run Experiment 2: Generated Token Visual Attention Tracking")
    parser.add_argument("--video-path", type=str, default=None, help="Path to video file")
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen3-VL-8B-Instruct", help="HF model ID")
    parser.add_argument("--output-dir", type=str, default="results/exp2", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda", help="Target device")
    parser.add_argument("--prompt-mode", type=str, default="direct", choices=["cot", "direct"], help="Prompting mode")
    parser.add_argument("--max-new-tokens", type=int, default=100, help="Max tokens to generate")
    parser.add_argument("--force-text-path", type=str, default=None, help="Path to text file containing target response to force decode")
    parser.add_argument("--plot-all-tokens", action="store_true", help="Plot heatmaps for all generated tokens instead of filtering")
    args = parser.parse_args()
    
    model, processor = load_model_and_processor(
        args.model_id,
        device=args.device,
        attn_implementation="eager"
    )
    
    forced_token_ids = None
    if args.force_text_path:
        if not os.path.exists(args.force_text_path):
            print(f"Error: Forced text file not found at {args.force_text_path}")
            return
        with open(args.force_text_path, "r", encoding="utf-8") as f:
            forced_text = f.read().strip()
        print(f"Loaded forced text ({len(forced_text)} chars). Tokenizing...")
        forced_token_ids = processor.tokenizer.encode(forced_text, add_special_tokens=False)
        print(f"Forced sequence contains {len(forced_token_ids)} tokens.")
    
    if args.video_path:
        instances = [get_associated_files(args.video_path)]
    else:
        # Default to a representative C2 bounce ball video
        default_dir = "videos/temporal/bounce_ball"
        if os.path.exists(default_dir):
            all_instances = find_video_files(default_dir)
            easy_instances = [
                inst for inst in all_instances 
                if inst["metadata"]["count"] == 2
            ]
            if easy_instances:
                # Prefer f0.5 and s0 if available
                matching_subset = [
                    inst for inst in easy_instances 
                    if inst["metadata"].get("seed") == 0 
                    and inst["metadata"].get("frequency") == 0.5
                ]
                if matching_subset:
                    instances = matching_subset[:1]
                else:
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
            if "bounce" in video_path.lower():
                raw_question = "How many times did the ball bounce?"
            elif "state" in video_path.lower() or "transition" in video_path.lower():
                raw_question = "How many color changes occurred in total?"
            else:
                raw_question = "How many times did the object flash?"
        else:
            with open(q_path, "r") as f:
                raw_question = f.read().strip()
                
        question_text = format_prompt_by_mode(raw_question, args.prompt_mode)
        # Prepare visual inputs at 2.0 FPS
        # Prefill \boxed{ in direct mode (if not teacher-forcing) to steer the model to output the count directly
        prefill_boxed = (args.prompt_mode == "direct" and not args.force_text_path)
        inputs = prepare_video_inputs(video_path, question_text, processor, device=args.device, fps=2.0, prefill_boxed=prefill_boxed)
        
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        print(f"\nProcessing Generated Token Attention for {video_name} (Mode: {args.prompt_mode.upper()}, 2.0 FPS)...")
        
        try:
            # Configure a temporary directory for offloaded attention weights
            temp_dir = os.path.join(args.output_dir, f"temp_attn_{video_name}")
            os.makedirs(temp_dir, exist_ok=True)
            attn_capture_state.temp_dir = temp_dir
            
            results = run_generated_token_attention_rollout(
                model, 
                inputs, 
                processor, 
                max_new_tokens=args.max_new_tokens,
                forced_token_ids=forced_token_ids
            )
            results["video_name"] = os.path.basename(video_path)
            results["prompt"] = question_text
            
            # Save temporal attention rollout heatmaps (no upscaling)
            for target_layer in [-1, -2]:
                out_img = os.path.join(args.output_dir, f"{video_name}_{args.prompt_mode}_layer{target_layer}_temporal_attention_shift.png")
                plot_generated_token_temporal_heatmap(results, out_img, target_layer=target_layer, duration=duration)
                print(f"    Saved temporal rollout shift plot for Layer {target_layer} to {out_img}")
                
            # Plot spatial visual rollout heatmaps overlaid on actual video frames
            tokens = results["generated_tokens"]
            print(f"    Generated {len(tokens)} tokens in total.")
            
            # Print the full generated answer
            full_gen = "".join(tokens[1:])
            if args.prompt_mode == "direct" and not args.force_text_path:
                full_gen = "\\boxed{" + full_gen
            print(f"\n    === Generated Response for {video_name} ({args.prompt_mode.upper()}) ===")
            print(f"    {full_gen.strip()}")
            print(f"    ==================================================\n")
            
            # We plot the heatmap for selected tokens (excluding <prefill> and stop tokens)
            target_indices = []
            for idx in range(1, len(tokens)):
                if tokens[idx] not in ["<|im_end|>", "<|endoftext|>"]:
                    if args.plot_all_tokens:
                        target_indices.append(idx)
                    else:
                        token_lower = tokens[idx].lower()
                        # Match digits
                        has_digit = any(char.isdigit() for char in token_lower)
                        # Match key terms
                        keywords = ["bounce", "touch", "contact", "first", "second", "stationary", "movement", "twice", "boxed"]
                        has_keyword = any(kw in token_lower for kw in keywords)
                        if has_digit or has_keyword:
                            target_indices.append(idx)
            
            print(f"    Filtering tokens for spatial heatmap overlays (plotting key concepts / numbers)...")
            print(f"    Plotting {len(target_indices)} out of {len(tokens) - 1} generated tokens.")
                
            for token_idx in target_indices:
                safe_token_name = "".join([c if c.isalnum() else "_" for c in tokens[token_idx]]).strip("_")
                if not safe_token_name:
                    safe_token_name = f"idx{token_idx}"
                out_img = os.path.join(args.output_dir, f"{video_name}_{args.prompt_mode}_token{token_idx}_{safe_token_name}_spatial_heatmap.png")
                plot_spatial_attention_for_token(results, token_idx, video_path, out_img, target_layer=-2, duration=duration)
                print(f"    Saved spatial visual rollout heatmap overlay for token [{token_idx}] '{tokens[token_idx]}' to {out_img}")
                
            # Generate the specialized paper-style rollout analysis plots for the final answer token
            if target_indices:
                final_token_idx = target_indices[-1]
                trace_path = inst.get("trace_path", None)
                
                # 1. Layer-by-frame heatmap (M_{l,t})
                out_img_layer_frame = os.path.join(args.output_dir, f"{video_name}_{args.prompt_mode}_layer_by_frame_rollout.png")
                plot_layer_by_frame_heatmap(results, final_token_idx, out_img_layer_frame, duration=duration)
                print(f"    Saved layer-by-frame rollout heatmap to {out_img_layer_frame}")
                
                # 2. Frame-level rollout curve with event times
                out_img_curve = os.path.join(args.output_dir, f"{video_name}_{args.prompt_mode}_rollout_curve_layer-2.png")
                plot_frame_level_rollout_curve(results, final_token_idx, trace_path, out_img_curve, target_layer=-2, duration=duration)
                print(f"    Saved frame-level rollout curve to {out_img_curve}")
                
                # 3. Raw vs rollout comparison with event times
                out_img_comp = os.path.join(args.output_dir, f"{video_name}_{args.prompt_mode}_raw_vs_rollout_layer-2.png")
                plot_raw_vs_rollout_comparison(results, final_token_idx, trace_path, out_img_comp, target_layer=-2, duration=duration)
                print(f"    Saved raw vs rollout comparison to {out_img_comp}")
                
            # Save results dictionary to JSON
            summary_results = {
                "video_name": results["video_name"],
                "prompt": results["prompt"],
                "generated_tokens": results["generated_tokens"],
                "T": results["T"],
                "T_out": results["T_out"],
                "H_out": results["H_out"],
                "W_out": results["W_out"],
                "num_layers": results["num_layers"]
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
