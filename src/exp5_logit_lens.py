import os
import argparse
import json
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from src.utils.model_helpers import load_model_and_processor, prepare_video_inputs, get_associated_files, find_video_files, format_prompt_by_mode

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
            probs = F.softmax(logits, dim=-1).float().cpu().numpy()
            
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
    Plots the logit lens probabilities across the model's layers with side-by-side
    linear and log-scale subplots.
    """
    layers = results["num_layers"]
    correct_token = results["correct_token"]
    correct_probs = results["correct_probs"]
    alt_probs = results["alternative_probs"]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = ['orange', 'red', 'blue', 'purple']
    
    # Left subplot: Linear Scale
    axes[0].plot(range(layers + 1), correct_probs, label=f"Correct ('{correct_token}')", color='green', linewidth=2.5, marker='o')
    for idx, (token_str, probs) in enumerate(alt_probs.items()):
        color = colors[idx % len(colors)]
        axes[0].plot(range(layers + 1), probs, label=f"Alternative ('{token_str}')", color=color, linestyle='--', marker='x')
    axes[0].set_xlabel('Layer Index')
    axes[0].set_ylabel('Vocabulary Probability')
    axes[0].set_title('Linear Scale')
    axes[0].grid(True)
    axes[0].legend()
    
    # Right subplot: Log Scale
    axes[1].plot(range(layers + 1), correct_probs, label=f"Correct ('{correct_token}')", color='green', linewidth=2.5, marker='o')
    for idx, (token_str, probs) in enumerate(alt_probs.items()):
        color = colors[idx % len(colors)]
        axes[1].plot(range(layers + 1), probs, label=f"Alternative ('{token_str}')", color=color, linestyle='--', marker='x')
    axes[1].set_yscale('log')
    # Set reasonable bounds for log scale (e.g. 1e-6 to 1.5) to avoid -inf issues
    axes[1].set_ylim([1e-6, 1.5])
    axes[1].set_xlabel('Layer Index')
    axes[1].set_ylabel('Vocabulary Probability (Log)')
    axes[1].set_title('Log Scale (Lower bound 1e-6)')
    axes[1].grid(True, which='both', linestyle=':', alpha=0.5)
    axes[1].legend()
    
    fig.suptitle(title, fontsize=12, fontweight='bold')
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
    parser.add_argument("--prompt-mode", type=str, default="both", choices=["cot", "direct", "both"], help="Prompting mode (cot, direct, or both)")
    args = parser.parse_args()
    
    if args.video_path is not None and args.video_dir is not None:
        parser.error("Only one of --video-path or --video-dir can be specified.")
        
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
            
        # Collect target video instances
        instances = []
        if is_single_video:
            instances.append((get_associated_files(target_path), "single_video"))
            print(f"\nTargeting single video in domain '{domain_name}': {target_path}")
        else:
            all_instances = find_video_files(target_path)
            
            # Select Cohorts A (Easy) and B (Trap)
            cohort_A = [inst for inst in all_instances if inst["metadata"]["count"] is not None and inst["metadata"]["count"] <= 3 and inst["metadata"]["frequency"] is not None and inst["metadata"]["frequency"] <= 1.0]
            cohort_B = [inst for inst in all_instances if inst["metadata"]["count"] is not None and inst["metadata"]["count"] >= 5 and inst["metadata"]["frequency"] is not None and inst["metadata"]["frequency"] <= 1.0]
            
            import random
            random.Random(42).shuffle(cohort_A)
            random.Random(42).shuffle(cohort_B)
            
            if cohort_A:
                instances.append((cohort_A[0], "cohort_A"))
            if cohort_B:
                instances.append((cohort_B[0], "cohort_B"))
                
            if not instances:
                print(f"Error: No videos resolved for domain '{domain_name}'.")
                continue
                
            print(f"\nResolved diagnostic instances for domain '{domain_name}':")
            for inst, label in instances:
                print(f"  - {label}: {os.path.basename(inst['video_path'])}")

        # Loop over prompt modes
        modes_to_run = ["cot", "direct"] if args.prompt_mode == "both" else [args.prompt_mode]
        for prompt_mode in modes_to_run:
            print(f"\n--- Running Prompt Mode: {prompt_mode.upper()} ---")
            output_dir = os.path.join(args.output_dir, domain_name, prompt_mode)
            os.makedirs(output_dir, exist_ok=True)
            
            for inst, cohort_label in instances:
                video_path = inst["video_path"]
                q_path = inst["question_path"]
                solution_path = inst["solution_path"]
                metadata = inst["metadata"]
                
                if not q_path or not solution_path:
                    print(f"Warning: Associated question or solution file not found for {video_path}")
                    continue
                    
                with open(q_path, "r") as f:
                    raw_question = f.read().strip()
                    
                question_text = format_prompt_by_mode(raw_question, prompt_mode)
                    
                with open(solution_path, "r") as f:
                    ground_truth = int(f.read().strip())
                    
                print(f"  Running Logit Lens ({cohort_label}) on: {os.path.basename(video_path)} (GT Count = {ground_truth})")
                
                inputs = prepare_video_inputs(video_path, question_text, processor, device=args.device, prefill_boxed=(prompt_mode == "direct"))
                
                correct_token_str = str(ground_truth)
                alternative_token_strs = [str(ground_truth - 1), str(ground_truth - 2), str(ground_truth + 1)]
                
                try:
                    results = run_logit_lens(model, inputs, processor, correct_token_str, alternative_token_strs)
                    results["video_name"] = os.path.basename(video_path)
                    results["cohort"] = cohort_label
                    results["prompt"] = question_text
                    
                    # Run text generation to verify what the model actually outputs
                    print("    Generating model's reasoning and count answer...")
                    with torch.no_grad():
                        generated_ids = model.generate(**inputs, max_new_tokens=512)
                        generated_ids = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)]
                        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                        if prompt_mode == "direct" and not generated_text.startswith("\\boxed{"):
                            generated_text = "\\boxed{" + generated_text
                    results["generated_response"] = generated_text
                    print(f"    Generated Response: {generated_text.strip().replace(chr(10), ' ')}")
                    
                    # Save Logit Lens Plot with cohort prefix
                    out_img = os.path.join(output_dir, f"{cohort_label}_{os.path.splitext(os.path.basename(video_path))[0]}_logit_lens.png")
                    plot_logit_lens(
                        results, out_img, 
                        f"Logit Lens Profile ({cohort_label}) for {os.path.basename(video_path)} (GT={ground_truth})"
                    )
                    
                    # Save JSON output with cohort prefix
                    out_json = os.path.join(output_dir, f"{cohort_label}_{os.path.splitext(os.path.basename(video_path))[0]}_logit_lens.json")
                    with open(out_json, "w") as f:
                        json.dump(results, f, indent=2)
                    print(f"Finished Logit Lens for {cohort_label} in {domain_name} ({prompt_mode})! Saved to {out_json}")
                except Exception as e:
                    print(f"  Error running Logit Lens: {e}")
                    import traceback
                    traceback.print_exc()
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

if __name__ == "__main__":
    main()
