import sys
import os
from transformers import AutoModelForImageTextToText

def main():
    model_id = "Qwen/Qwen3-VL-8B-Instruct"
    output_file = "qwen3_vl_architecture.txt"
    
    print(f"Loading model structure for {model_id} using meta device (no weights loaded, fast & memory-efficient)...")
    try:
        # device_map="meta" loads the shell of the model structure instantly without using any RAM/GPU memory
        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            device_map="meta",
            trust_remote_code=True
        )
    except Exception as e:
        print(f"Could not load using meta device ({e}). Trying to load on CPU...")
        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        
    print(f"Writing model architecture description to '{output_file}'...")
    with open(output_file, "w") as f:
        f.write(str(model))
        
    print(f"Success! Model architecture description written to '{os.path.abspath(output_file)}'.")

if __name__ == "__main__":
    main()
