import os
import re
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from qwen_vl_utils import process_vision_info

DEFAULT_MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"

def load_model_and_processor(model_id=DEFAULT_MODEL_ID, device="cuda", flash_attn=True, attn_implementation=None):
    """
    Loads Qwen3-VL model and processor.
    
    Args:
        model_id (str): Hugging Face repository or local path.
        device (str): target PyTorch device ('cuda', 'cpu', etc.)
        flash_attn (bool): whether to use flash_attention_2 if available.
        attn_implementation (str): target attention implementation ('flash_attention_2', 'sdpa', 'eager').
    """
    print(f"Loading processor for {model_id}...")
    processor = AutoProcessor.from_pretrained(model_id)
    
    print(f"Loading model {model_id} on {device}...")
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    
    if attn_implementation is None:
        # Check if flash_attn is installed and available
        has_flash_attn = False
        if flash_attn and torch.cuda.is_available():
            try:
                import flash_attn
                has_flash_attn = True
            except ImportError:
                print("flash_attn package is not installed. Defaulting to SDPA.")
                
        attn_implementation = "flash_attention_2" if has_flash_attn else "sdpa"
    
    try:
        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
            device_map=device
        )
    except (ImportError, ValueError) as e:
        if attn_implementation == "flash_attention_2":
            print(f"Failed loading with flash_attention_2 ({e}). Falling back to sdpa...")
            attn_implementation = "sdpa"
            model = AutoModelForImageTextToText.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
                attn_implementation=attn_implementation,
                device_map=device
            )
        else:
            raise e
    
    return model, processor


def prepare_video_inputs(video_path, question_text, processor, device="cuda", fps=2.0):
    """
    Prepares Hugging Face model inputs for a video and question.
    
    Args:
        video_path (str): path to video file.
        question_text (str): prompt/query for the model.
        processor: HF AutoProcessor instance.
        device (str): target device.
        fps (float/int): override for frame sampling rate (defaults to 2.0).
    """
    # Check for a functional video reader backend
    try:
        import torchvision.io as tv_io
        has_tv_read = hasattr(tv_io, "read_video")
    except ImportError:
        has_tv_read = False

    try:
        import decord
        has_decord = True
    except ImportError:
        has_decord = False

    if not has_tv_read and not has_decord:
        raise ImportError(
            "\n" + "="*80 + "\n"
            "ERROR: No functional video reader backend found for qwen_vl_utils.\n"
            "Your torchvision installation was compiled without video reading support (missing FFmpeg/PyAV),\n"
            "and the 'decord' package is not installed.\n\n"
            "To resolve this, please install one of the following on your GPU server:\n"
            "  pip install av\n"
            "or:\n"
            "  pip install decord\n"
            "  (Note: You can force a backend by running: export FORCE_QWENVL_VIDEO_READER=decord)\n" +
            "="*80 + "\n"
        )
    elif not has_tv_read and has_decord:
        if not os.environ.get("FORCE_QWENVL_VIDEO_READER"):
            print("torchvision is missing read_video support. Forcing decord backend...")
            os.environ["FORCE_QWENVL_VIDEO_READER"] = "decord"

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
        return_tensors="pt",
        do_sample_frames=False
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
    
    # Locate visual tokens (excluding newlines/special tokens if possible)
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
    
    hidden_states = outputs.hidden_states[layer_idx][0].clone() # shape: (seq_len, hidden_dim)
    
    # Free other hidden states layers from memory
    del outputs
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Try to extract the precise visual placeholder tokens (excluding any structure/newlines)
    video_pad_id = vocab.get("<|video_pad|>", None)
    image_pad_id = vocab.get("<|image_pad|>", None)
    if video_pad_id is None:
        video_pad_id = getattr(processor, "video_token_id", None)
    if image_pad_id is None:
        image_pad_id = getattr(processor, "image_token_id", None)
        
    visual_ids = [vid for vid in [video_pad_id, image_pad_id] if vid is not None]
    visual_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for vid in visual_ids:
        visual_mask = visual_mask | (input_ids == vid)
        
    # Restrict mask to the region between the vision start and end tokens
    in_vision_bounds = torch.zeros_like(input_ids, dtype=torch.bool)
    in_vision_bounds[start_idx:end_idx] = True
    visual_mask = visual_mask & in_vision_bounds
    
    visual_positions = visual_mask.nonzero(as_tuple=True)[0]
    
    # Calculate target grid size (considering temporal_patch_size=2 downsampling and spatial_merge_size=2)
    T_out = max(1, T // 2)
    H_out = H // 2
    W_out = W // 2
    expected_tokens = T_out * H_out * W_out
    
    if len(visual_positions) == expected_tokens:
        hidden_vision = hidden_states[visual_positions] # shape: (expected_tokens, hidden_dim)
        hidden_vision = hidden_vision.view(T_out, H_out * W_out, -1)
        temporal_trajectory = torch.mean(hidden_vision, dim=1).cpu().float().numpy() # (T_out, hidden_dim)
    else:
        # Fallback to slicing in case of unforeseen token mappings
        hidden_vision = hidden_states[start_idx:end_idx]
        num_tokens = hidden_vision.shape[0]
        # Distribute tokens as evenly as possible across T_out steps
        step_size = max(1, num_tokens // T_out)
        trajectories = []
        for t in range(T_out):
            start = t * step_size
            end = (t + 1) * step_size if t < T_out - 1 else num_tokens
            if start < num_tokens:
                trajectories.append(torch.mean(hidden_vision[start:end], dim=0))
            else:
                trajectories.append(torch.zeros(hidden_states.shape[-1], device=hidden_states.device))
        temporal_trajectory = torch.stack(trajectories).cpu().float().numpy()
        
    return temporal_trajectory

