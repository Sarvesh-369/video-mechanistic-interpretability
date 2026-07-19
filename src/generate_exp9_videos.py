import argparse
import os
import shutil
import json
import random
import sys
from pathlib import Path

# Add project root to path to resolve src imports
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.generate_custom_datasets import CustomBouncingBall


def main():
    parser = argparse.ArgumentParser(description="Generate control videos for Experiment 9 (Timing Regularity)")
    parser.add_argument("--out-dir", type=str, default="videos/exp9", help="Output directory")
    parser.add_argument("--seeds", type=int, default=10, help="Number of random seeds per cell")
    parser.add_argument("--duration", type=float, default=24.0, help="Total video duration in seconds")
    parser.add_argument("--span", type=float, default=24.0, help="Active bouncing span in seconds")
    args = parser.parse_args()
    
    out_dir = Path(args.out_dir)
    (out_dir / "questions").mkdir(parents=True, exist_ok=True)
    (out_dir / "solutions").mkdir(parents=True, exist_ok=True)
    (out_dir / "question_text").mkdir(parents=True, exist_ok=True)
    (out_dir / "reasoning_traces").mkdir(parents=True, exist_ok=True)
    
    counts = [3, 4, 5, 6, 7, 8]
    regularities = ["periodic", "irregular"]
    
    print("=========================================================")
    echo_msg = f"Generating Exp 9: Irregularity vs Regularity ({len(counts)} counts, {len(regularities)} modes, {args.seeds} seeds)"
    print(echo_msg)
    print("=========================================================")
    
    for count in counts:
        for regularity in regularities:
            for seed in range(args.seeds):
                base_name = f"exp9_n{count}_reg{regularity}_seed{seed}"
                
                # Paths
                final_video_path = out_dir / "questions" / f"{base_name}.mp4"
                final_json_path = out_dir / "questions" / f"{base_name}.json"
                
                if final_video_path.exists() and final_json_path.exists():
                    print(f"Skipping {base_name} (already exists).")
                    continue
                    
                print(f"Rendering: {base_name} (Count={count}, Reg={regularity}, Span={args.span}s)")
                
                scene = CustomBouncingBall(
                    count=count,
                    span=args.span,
                    position="early",
                    regularity=regularity,
                    duration=args.duration,
                    seed=seed
                )
                scene.render()
                
                expected_output = Path("manim_output/videos/480p15/CustomBouncingBall.mp4")
                if expected_output.exists():
                    shutil.move(str(expected_output), str(final_video_path))
                    
                    # Write solution
                    with open(out_dir / "solutions" / f"{base_name}.txt", "w") as f:
                        f.write(str(scene.answer))
                        
                    # Write question text
                    with open(out_dir / "question_text" / f"{base_name}.txt", "w") as f:
                        f.write(scene.question_text)
                        
                    # Write reasoning traces
                    with open(out_dir / "reasoning_traces" / f"{base_name}.txt", "w") as f:
                        f.write("\n".join(scene.reasoning_trace))
                        
                    # Write metadata json
                    meta = {
                        "experiment": "9",
                        "count": count,
                        "span": args.span,
                        "position": "early",
                        "regularity": regularity,
                        "duration": args.duration,
                        "seed": seed,
                        "bounce_times": scene.bounce_times
                    }
                    with open(final_json_path, "w") as f:
                        json.dump(meta, f, indent=2)
                        
                    print(f"✓ Saved files successfully for {base_name}")
                else:
                    print(f"Error: Render did not produce expected file for {base_name}")
                    
                # Clean up temporary Manim folder
                if os.path.exists("manim_output"):
                    shutil.rmtree("manim_output")

if __name__ == "__main__":
    main()
