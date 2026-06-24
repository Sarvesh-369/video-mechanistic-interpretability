import os
import torch
import numpy as np
from src.exp6_generated_token_attention import run_generated_token_attention_rollout
from src.utils.model_helpers import load_model_and_processor, prepare_video_inputs, format_prompt_by_mode

def inspect():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load model
    model, processor = load_model_and_processor(
        "Qwen/Qwen3-VL-8B-Instruct",
        device=device,
        attn_implementation="eager"
    )
    
    # Target video
    video_path = "videos/temporal/bounce_ball/sweep_count_bounces_c2_f0.5_s0_d24.0_count_bounces.mp4"
    if not os.path.exists(video_path):
        print(f"Video not found at {video_path}")
        return
        
    question_text = format_prompt_by_mode("How many times did the ball bounce?", "cot")
    inputs = prepare_video_inputs(video_path, question_text, processor, device=device, fps=2.0, prefill_boxed=False)
    
    # Load forced text
    forced_text_path = "scratch/forced_text.txt"
    with open(forced_text_path, "r", encoding="utf-8") as f:
        forced_text = f.read().strip()
    forced_token_ids = processor.tokenizer.encode(forced_text, add_special_tokens=False)
    
    # Run rollout for first 30 tokens to get statistics
    results = run_generated_token_attention_rollout(
        model, 
        inputs, 
        processor, 
        max_new_tokens=30,
        forced_token_ids=forced_token_ids[:30]
    )
    
    tokens = results["generated_tokens"]
    attentions = np.array(results["attentions"]) # (steps, layers, T, H, W)
    num_layers = results["num_layers"]
    layer_idx = num_layers - 2 # Layer 34
    
    print("\n--- Quantitative Attention Analysis (Layer 34) ---")
    for idx in range(1, len(tokens)):
        token_str = tokens[idx]
        token_attn = attentions[idx][layer_idx] # (T, H, W)
        flat_attn = token_attn.flatten()
        
        # Calculate statistics
        total = np.sum(flat_attn)
        mean_val = np.mean(flat_attn)
        max_val = np.max(flat_attn)
        min_val = np.min(flat_attn)
        std_val = np.std(flat_attn)
        
        # Percent of attention on top 5 spatial-temporal locations
        sorted_indices = np.argsort(flat_attn)[::-1]
        top5_pct = np.sum(flat_attn[sorted_indices[:5]]) / (total + 1e-9) * 100
        
        # Check coordinates of top location
        top1_idx = sorted_indices[0]
        t_idx = top1_idx // (results["H_out"] * results["W_out"])
        rem = top1_idx % (results["H_out"] * results["W_out"])
        h_idx = rem // results["W_out"]
        w_idx = rem % results["W_out"]
        
        print(f"Token {idx:2d} | {repr(token_str):12s} | Max: {max_val:.4e} | Min: {min_val:.4e} | Top5 Pct: {top5_pct:.1f}% | Top1 at T={t_idx}, H={h_idx}, W={w_idx}")

if __name__ == "__main__":
    inspect()
