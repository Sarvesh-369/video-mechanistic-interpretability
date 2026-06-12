import os
import re
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from qwen_vl_utils import process_vision_info

DEFAULT_MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"

def load_model_and_processor(model_id=DEFAULT_MODEL_ID, device="cuda", flash_attn=True):
    """
    Loads Qwen3-VL model and processor.
    
    Args:
        model_id (str): Hugging Face repository or local path.
        device (str): target PyTorch device ('cuda', 'cpu', etc.)
        flash_attn (bool): whether to use flash_attention_2 if available.
    """
    print(f"Loading processor for {model_id}...")
    processor = AutoProcessor.from_pretrained(model_id)
    
    print(f"Loading model {model_id} on {device}...")
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    
    attn_implementation = "flash_attention_2" if flash_attn and torch.cuda.is_available() else "sdpa"
    
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        attn_implementation=attn_implementation,
        device_map=device
    )
    
    return model, processor


def prepare_video_inputs(video_path, question_text, processor, device="cuda", fps=None):
    """
    Prepares Hugging Face model inputs for a video and question.
    
    Args:
        video_path (str): path to video file.
        question_text (str): prompt/query for the model.
        processor: HF AutoProcessor instance.
        device (str): target device.
        fps (float/int): optional override for frame sampling rate.
    """
    content = []
    video_item = {"type": "video", "video": os.path.abspath(video_path)}
    if fps is not None:
        video_item["fps"] = fps
        
    content.append(video_item)
    content.append({"type": "text", "text": question_text})
    
    messages = [
        {
            "role": "user",
            "content": content
        }
    ]
    
    # Formulate chat template prompt
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # Process images and videos via qwen_vl_utils helper
    image_inputs, video_inputs = process_vision_info(messages)
    
    # Get tokenized inputs
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    )
    
    # Move tensor inputs to correct device
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    return inputs


def parse_video_filename(filename):
    """
    Parses metadata variables from synthetic video filenames like:
    sweep_count_blinks_c10_f0.5_s0_d24.0_count_blinks.mp4
    
    Returns:
        dict: Dict containing parsed parameters (count, frequency, seed, duration).
    """
    basename = os.path.basename(filename)
    
    # Regular expressions to match count (c), frequency (f), seed (s), duration (d)
    count_match = re.search(r'_c(\d+)_', basename)
    freq_match = re.search(r'_f(\d+\.?\d*)_', basename)
    seed_match = re.search(r'_s(\d+)_', basename)
    dur_match = re.search(r'_d(\d+\.?\d*)_', basename)
    
    return {
        "count": int(count_match.group(1)) if count_match else None,
        "frequency": float(freq_match.group(1)) if freq_match else None,
        "seed": int(seed_match.group(1)) if seed_match else None,
        "duration": float(dur_match.group(1)) if dur_match else None,
    }


def get_associated_files(video_path):
    """
    Given a single video path, resolves its corresponding question, reasoning trace, 
    and solution text files using relative directory structure.
    """
    video_dir = os.path.dirname(video_path) # e.g. parent/questions
    parent_dir = os.path.dirname(video_dir) # e.g. parent
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    
    question_path = os.path.join(parent_dir, "question_text", base_name + ".txt")
    trace_path = os.path.join(parent_dir, "reasoning_traces", base_name + ".txt")
    solution_path = os.path.join(parent_dir, "solutions", base_name + ".txt")
    
    return {
        "video_path": video_path,
        "question_path": question_path if os.path.exists(question_path) else None,
        "trace_path": trace_path if os.path.exists(trace_path) else None,
        "solution_path": solution_path if os.path.exists(solution_path) else None,
        "metadata": parse_video_filename(video_path)
    }


def find_video_files(video_dir):
    """
    Scans a task directory for video questions and associated solutions and reasoning traces.
    
    Returns:
        list of dicts: List of instances containing paths to video, question, reasoning trace, and solution.
    """
    videos_path = os.path.join(video_dir, "questions")
    if not os.path.exists(videos_path):
        return []
        
    instances = []
    for filename in sorted(os.listdir(videos_path)):
        if filename.endswith(".mp4"):
            video_file = os.path.join(videos_path, filename)
            base_name = filename.rsplit(".", 1)[0]
            
            question_file = os.path.join(video_dir, "question_text", base_name + ".txt")
            trace_file = os.path.join(video_dir, "reasoning_traces", base_name + ".txt")
            solution_file = os.path.join(video_dir, "solutions", base_name + ".txt")
            
            instances.append({
                "video_path": video_file,
                "question_path": question_file if os.path.exists(question_file) else None,
                "trace_path": trace_file if os.path.exists(trace_file) else None,
                "solution_path": solution_file if os.path.exists(solution_file) else None,
                "metadata": parse_video_filename(filename)
            })
            
    return instances


def extract_representation_trajectory(model, inputs, processor, layer_idx=-1):
    """
    Extracts the intermediate hidden states of the visual tokens at a specified layer.
    
    Args:
        model: Loaded Qwen3-VL model.
        inputs: Tokenized input dictionary.
        processor: Loaded AutoProcessor.
        layer_idx (int): index of the transformer layer to extract from.
    
    Returns:
        np.ndarray: shape (T, hidden_dim) representing spatial-mean pooled hidden states per frame.
    """
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        
    input_ids = inputs["input_ids"][0]
    
    # Locate visual tokens
    vocab = processor.tokenizer.get_vocab()
    vision_start_id = vocab.get("<|vision_start|>", 151652)
    vision_end_id = vocab.get("<|vision_end|>", 151653)
    
    start_pos = (input_ids == vision_start_id).nonzero(as_tuple=True)[0]
    end_pos = (input_ids == vision_end_id).nonzero(as_tuple=True)[0]
    
    if len(start_pos) == 0 or len(end_pos) == 0:
        raise ValueError("Could not locate visual start/end tokens.")
        
    start_idx = start_pos[0].item() + 1
    end_idx = end_pos[0].item()
    
    # Get grid info
    video_grid_thw = inputs["video_grid_thw"][0].cpu().numpy()
    T, H, W = int(video_grid_thw[0]), int(video_grid_thw[1]), int(video_grid_thw[2])
    
    # Retrieve hidden states for target layer
    # outputs.hidden_states is a tuple of length num_layers + 1
    hidden_states = outputs.hidden_states[layer_idx][0] # shape: (seq_len, hidden_dim)
    
    # Extract visual token representations
    hidden_vision = hidden_states[start_idx:end_idx] # shape: (T*H*W, hidden_dim)
    
    # Reshape to (T, H*W, hidden_dim) and perform spatial mean pooling
    hidden_dim = hidden_vision.shape[-1]
    hidden_vision = hidden_vision.view(T, H * W, hidden_dim)
    
    # Mean pool over spatial patch dimension
    temporal_trajectory = torch.mean(hidden_vision, dim=1).cpu().float().numpy() # (T, hidden_dim)
    
    return temporal_trajectory

