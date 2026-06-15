import os
import argparse
import json
import torch
import re
from src.utils.model_helpers import load_model_and_processor, get_associated_files, find_video_files, format_prompt_by_mode
from qwen_vl_utils import process_vision_info

def prepare_inputs_with_ablation(video_path, question_text, processor, device, config):
    """
    Prepares inputs with custom spatial-temporal preprocessing configurations.
    """
    content = []
    video_item = {
        "type": "video",
        "video": os.path.abspath(video_path)
    }
    
    # Apply overrides
    if config.get("fps") is not None:
        video_item["fps"] = config["fps"]
    if config.get("min_pixels") is not None:
        video_item["min_pixels"] = config["min_pixels"]
    if config.get("max_pixels") is not None:
        video_item["max_pixels"] = config["max_pixels"]
        
    content.append(video_item)
    content.append({"type": "text", "text": question_text})
    
    messages = [
        {
            "role": "user",
            "content": content
        }
    ]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        do_sample_frames=False
    )
    
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    return inputs

def parse_answer(output_text):
    """
    Parses the numeric count out of the LaTeX boxed format like \boxed{5}.
    Falls back to looking for a single isolated number in the text if no boxed value is found.
    """
    match = re.search(r'\\boxed\{(\d+)\}', output_text)
    if match:
        return int(match.group(1))
    
    # Fallback for direct prompt mode where model often outputs only the number
    digits = re.findall(r'\b\d+\b', output_text.strip())
    if len(digits) == 1:
        return int(digits[0])
        
    return None

def main():
    parser = argparse.ArgumentParser(description="Run Experiment 4: Preprocessing Ablation Analysis")
    parser.add_argument("--video-path", type=str, default=None, help="Path to a single video file")
    parser.add_argument("--video-dir", type=str, default=None, help="Path to a directory containing video dataset")
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen3-VL-8B-Instruct", help="Hugging Face model ID")
    parser.add_argument("--output-dir", type=str, default="results/exp4", help="Output directory")
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
            
        # Preprocessing configs to ablate
        configs = {
            "Baseline": {
                # Uses processor defaults
            },
            "High_Temporal_Resolution": {
                "fps": 4.0
            },
            "High_Spatial_Resolution": {
                "min_pixels": 512 * 512,
                "max_pixels": 602112
            },
            "High_Temporal_And_Spatial": {
                "fps": 4.0,
                "min_pixels": 512 * 512,
                "max_pixels": 602112
            }
        }
        
        # Collect instances
        instances = []
        if is_single_video:
            instances.append(get_associated_files(target_path))
            print(f"\nTargeting single video in domain '{domain_name}': {target_path}")
        else:
            # Find boundary cases in directory: c in [4, 6] and f <= 1.0
            all_instances = find_video_files(target_path)
            boundary_instances = [
                inst for inst in all_instances 
                if inst["metadata"]["count"] is not None 
                and 4 <= inst["metadata"]["count"] <= 6
                and inst["metadata"]["frequency"] is not None
                and inst["metadata"]["frequency"] <= 1.0
            ]
            
            # Fallback to any videos if boundary instances not found
            if not boundary_instances:
                boundary_instances = all_instances[:8]
                
            instances = boundary_instances[:10]
            print(f"\nTargeting boundary cohort directory for domain '{domain_name}': {target_path} (Found {len(instances)} boundary instances)")

        # Loop over prompt modes
        modes_to_run = ["cot", "direct"] if args.prompt_mode == "both" else [args.prompt_mode]
        for prompt_mode in modes_to_run:
            print(f"\n--- Running Prompt Mode: {prompt_mode.upper()} ---")
            output_dir = os.path.join(args.output_dir, domain_name, prompt_mode)
            os.makedirs(output_dir, exist_ok=True)
            
            results_summary = []
            
            for idx, inst in enumerate(instances):
                video_path = inst["video_path"]
                q_path = inst["question_path"]
                solution_path = inst["solution_path"]
                metadata = inst["metadata"]
                
                if not q_path or not solution_path:
                    continue
                    
                with open(q_path, "r") as f:
                    raw_question = f.read().strip()
                    
                question_text = format_prompt_by_mode(raw_question, prompt_mode)
                    
                with open(solution_path, "r") as f:
                    ground_truth = int(f.read().strip())
                    
                print(f"  Video [{idx+1}/{len(instances)}]: {os.path.basename(video_path)} (GT Count = {ground_truth})")
                
                instance_results = {
                    "video_name": os.path.basename(video_path),
                    "metadata": metadata,
                    "ground_truth": ground_truth,
                    "prompt": question_text,
                    "predictions": {}
                }
                
                for name, config in configs.items():
                    print(f"    Running under config: {name}...")
                    
                    inputs = prepare_inputs_with_ablation(
                        video_path, question_text, processor, device=args.device, config=config
                    )
                    
                    try:
                        with torch.no_grad():
                            output_ids = model.generate(**inputs, max_new_tokens=1024)
                        
                        input_len = inputs["input_ids"].shape[1]
                        response = processor.decode(output_ids[0][input_len:], skip_special_tokens=True)
                        
                        predicted_count = parse_answer(response)
                        is_correct = (predicted_count == ground_truth)
                        
                        instance_results["predictions"][name] = {
                            "text_response": response.strip(),
                            "parsed_count": predicted_count,
                            "correct": is_correct
                        }
                        
                        print(f"      [{name}] Predicted: {predicted_count} | Correct: {is_correct}")
                    except Exception as e:
                        print(f"      Error under {name}: {e}")
                        instance_results["predictions"][name] = {
                            "error": str(e),
                            "correct": False
                        }
                        del e
                    finally:
                        if "inputs" in locals() and inputs is not None:
                            for k in list(inputs.keys()):
                                inputs[k] = None
                            inputs = None
                        if "output_ids" in locals():
                            output_ids = None
                        import gc
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        
                results_summary.append(instance_results)
                
            if not results_summary:
                continue
                
            # Print accuracy summary
            print(f"\n--- ABLATION RESULTS SUMMARY FOR '{domain_name}' ({prompt_mode}) ---")
            config_success_counts = {name: 0 for name in configs}
            total_valid = len(results_summary)
            
            for r in results_summary:
                for name in configs:
                    if r["predictions"].get(name, {}).get("correct", False):
                        config_success_counts[name] += 1
                        
            for name, success_count in config_success_counts.items():
                acc = success_count / total_valid if total_valid > 0 else 0
                print(f"Configuration [{name}] Accuracy: {acc:.4f} ({success_count}/{total_valid})")
                
            # Save results to JSON
            out_json = os.path.join(output_dir, "ablation_evaluation_results.json")
            with open(out_json, "w") as f:
                json.dump({
                    "config_accuracies": {k: v / total_valid for k, v in config_success_counts.items()},
                    "details": results_summary
                }, f, indent=2)
            print(f"Finished Experiment 4 for {domain_name} ({prompt_mode})! Summary saved to {out_json}")

if __name__ == "__main__":
    main()
