import argparse
import os
import json
import re
import torch
import sys
from pathlib import Path

# Add project root to path to resolve src imports
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.utils.model_helpers import load_model_and_processor, format_prompt_by_mode, prepare_video_inputs


def extract_count_from_response(text):
    """
    Extracts the number inside \boxed{} if present, otherwise finds any standalone digit.
    """
    boxed_match = re.search(r'\\boxed\{(\d+)\}', text)
    if boxed_match:
        return int(boxed_match.group(1))
    numbers = re.findall(r'\b\d+\b', text)
    if numbers:
        return int(numbers[0])
    return -1

def run_vlm_inference(model, processor, inputs, max_new_tokens=64):
    """
    Runs text generation using HF model and processor.
    """
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
        
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
    ]
    
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    return output_text

def run_vllm_api_inference(video_path, question_text, vllm_url, model_id="Qwen/Qwen3-VL-8B-Instruct", fps=2.0):
    """
    Runs inference by sending request to a hosted vLLM OpenAI-compatible endpoint.
    """
    import requests
    headers = {"Content-Type": "application/json"}
    
    content = []
    if video_path is not None:
        abs_video_path = os.path.abspath(video_path)
        video_item = {
            "type": "video_url",
            "video_url": {
                "url": f"file://{abs_video_path}"
            }
        }
        content.append(video_item)
        
    content.append({
        "type": "text",
        "text": question_text
    })
    
    messages = [
        {
            "role": "user",
            "content": content
        }
    ]
    
    payload = {
        "model": model_id,
        "messages": messages,
        "max_tokens": 128,
        "temperature": 0.0
    }
    
    try:
        response = requests.post(f"{vllm_url}/chat/completions", headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"  vLLM API Error: {e}")
        return ""

def main(vllm_url_override=None):
    parser = argparse.ArgumentParser(description="Evaluate Experiment 9 (Timing Regularity)")
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen3-VL-8B-Instruct", help="Hugging Face model ID")
    parser.add_argument("--device", type=str, default="cuda", help="Target device")
    parser.add_argument("--prompt-mode", type=str, default="cot", choices=["cot", "direct"], help="Prompting mode")
    parser.add_argument("--data-dir", type=str, default="videos/exp9", help="Path to generated dataset")
    parser.add_argument("--output-dir", type=str, default="results/new_results", help="Output directory")
    parser.add_argument("--vllm-url", type=str, default=None, help="If provided, routes requests to hosted vLLM endpoint")
    args = parser.parse_args()
    
    # Allow overriding from python wrappers
    vllm_url = vllm_url_override if vllm_url_override is not None else args.vllm_url
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    questions_dir = Path(args.data_dir) / "questions"
    if not questions_dir.exists():
        print(f"Error: Questions directory not found at {questions_dir}. Please run generate_exp9_videos.py first.")
        return
        
    exp_jsons = sorted(list(questions_dir.glob("*.json")))
    if not exp_jsons:
        print(f"No metadata files found in {questions_dir}")
        return
        
    print(f"Loaded {len(exp_jsons)} metadata files for Experiment 9 evaluation")
    
    # Load VLM model & processor or setup API routing
    if vllm_url:
        print(f"Using hosted vLLM endpoint: {vllm_url}")
        model, processor = None, None
    else:
        model, processor = load_model_and_processor(args.model_id, device=args.device)
        
    results = []
    
    for q_json in exp_jsons:
        with open(q_json, "r") as f:
            meta = json.load(f)
            
        base_name = q_json.stem
        video_path = str(questions_dir / f"{base_name}.mp4")
        
        count = meta["count"]
        regularity = meta["regularity"]
        seed = meta["seed"]
        
        print(f"\nEvaluating {base_name}: Count={count}, Reg={regularity}")
        
        try:
            question_text = format_prompt_by_mode("How many times did the ball bounce?", args.prompt_mode)
            if vllm_url:
                response = run_vllm_api_inference(video_path, question_text, vllm_url, args.model_id, fps=2.0)
            else:
                inputs = prepare_video_inputs(video_path, question_text, processor, device=args.device, fps=2.0)
                response = run_vlm_inference(model, processor, inputs)
                
            pred_count = extract_count_from_response(response)
            
            results.append({
                "base_name": base_name,
                "count": count,
                "regularity": regularity,
                "seed": seed,
                "response": response,
                "pred_count": pred_count,
                "correct": (pred_count == count)
            })
            print(f"  Output: {repr(response.strip())} -> Extracted: {pred_count} (GT: {count})")
        except Exception as e:
            print(f"  Error processing {base_name}: {e}")
            
    out_path = Path(args.output_dir) / "exp9_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Saved Experiment 9 results to {out_path}")

if __name__ == "__main__":
    main()
