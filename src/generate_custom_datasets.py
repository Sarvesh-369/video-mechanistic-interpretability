from manim import *
import argparse
import os
import shutil
import random
import numpy as np
from pathlib import Path
import json

# Configure Manim
config.media_dir = "manim_output"
config.verbosity = "WARNING"
config.pixel_height = 480  # Optimized resolution to run quickly
config.pixel_width = 854
config.frame_rate = 15     # Optimized frame rate to minimize file sizes and rendering times
config.preview = False

class CustomBouncingBall(Scene):
    """
    A custom scene showing a ball bouncing between two walls under specific experimental controls.
    """
    def __init__(self, count=4, span=8.0, position="mid", regularity="periodic", duration=24.0, seed=0, **kwargs):
        super().__init__(**kwargs)
        self.count = count
        self.span = span
        self.position = position
        self.regularity = regularity
        self.video_duration = duration
        self.seed = seed
        
        self.reasoning_trace = []
        self.scene_events = []
        self.bounce_times = []
        
        # Randomization
        random.seed(self.seed)
        np.random.seed(self.seed)

        # Constant styling/params matching reference code
        self.wall_distance = 3.0
        self.ball_radius = 0.4
        self.ball_color = RED
        self.wall_color = GREY
        self.bg_color = BLACK
        self.start_direction_right = random.choice([True, False])
        
    def log_event(self, description):
        current_time = self.renderer.time
        self.scene_events.append({
            'time': current_time,
            'description': description
        })

    def format_time(self, seconds):
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        frac = int((seconds % 1) * 100)
        return f"{mins}:{secs:02d}.{frac:02d}"

    def construct(self):
        self.camera.background_color = self.bg_color
        
        # Setup Walls
        left_bound = -self.wall_distance
        right_bound = self.wall_distance
        wall_width = 0.2
        valid_min_x = left_bound + (wall_width / 2) + self.ball_radius
        valid_max_x = right_bound - (wall_width / 2) - self.ball_radius
        full_distance = valid_max_x - valid_min_x
        
        left_wall = Rectangle(width=wall_width, height=4, color=self.wall_color, fill_opacity=1)
        left_wall.move_to([left_bound, 0, 0])
        right_wall = Rectangle(width=wall_width, height=4, color=self.wall_color, fill_opacity=1)
        right_wall.move_to([right_bound, 0, 0])
        self.add(left_wall, right_wall)
        
        # Setup Ball starting position (centered)
        start_x = 0.0
        ball = Circle(radius=self.ball_radius, color=self.ball_color, fill_opacity=1)
        ball.move_to([start_x, 0, 0])
        self.add(ball)
        
        self.log_event(f"Ball appears at x={start_x:.2f}")
        
        # Determine temporal bounds
        if self.position == "early":
            t_start = 0.0
        elif self.position == "late":
            t_start = self.video_duration - self.span
        else: # "mid"
            t_start = (self.video_duration - self.span) / 2.0
            
        t_end = t_start + self.span
        
        # 1. Delay before active span
        if t_start > 0:
            self.wait(t_start)
            
        # 2. Compute segment durations based on regularity
        if self.count > 0:
            if self.regularity == "periodic":
                segment_times = [self.span / self.count] * self.count
            elif self.regularity == "jittered":
                # Jitter periodic times slightly (up to +/- 20%)
                base_time = self.span / self.count
                segment_times = []
                for _ in range(self.count):
                    jitter = random.uniform(-0.2, 0.2) * base_time
                    segment_times.append(base_time + jitter)
                # Re-normalize to sum exactly to span
                scale = self.span / sum(segment_times)
                segment_times = [t * scale for t in segment_times]
            else: # "irregular"
                # Generate highly uneven durations
                weights = [random.uniform(0.1, 1.0) for _ in range(self.count)]
                scale = self.span / sum(weights)
                segment_times = [w * scale for w in weights]
        else:
            segment_times = []
            
        # 3. Active Bouncing Loop
        elapsed_in_span = 0.0
        current_x = start_x
        moving_right = self.start_direction_right
        bounce_count = 0
        
        for k in range(self.count):
            seg_time = segment_times[k]
            
            # Target for this bounce segment
            target_x = valid_max_x if moving_right else valid_min_x
            local_vector = [target_x - current_x, 0, 0]
            
            # Animate the movement
            self.play(ball.animate.shift(local_vector), run_time=seg_time, rate_func=linear)
            
            bounce_count += 1
            current_t = self.renderer.time
            self.bounce_times.append(current_t)
            self.log_event(f"Ball hits {'Right' if moving_right else 'Left'} Wall (Bounce #{bounce_count})")
            
            # Toggle direction and update state
            moving_right = not moving_right
            current_x = target_x
            elapsed_in_span += seg_time
            
        # 4. Wait for remaining duration
        rem_time = self.video_duration - self.renderer.time
        if rem_time > 0:
            self.wait(rem_time)
            
        # 5. Build solution and reasoning trace
        self.answer = bounce_count
        self.question_text = "How many times did the ball bounce?"
        
        self.reasoning_trace.append("### Step 1: Parse Video Events")
        for idx, t_val in enumerate(self.bounce_times):
            self.reasoning_trace.append(f"- Event #{idx+1} detected at {self.format_time(t_val)}s ({t_val:.2f}s)")
            
        self.reasoning_trace.append("")
        self.reasoning_trace.append("### Step 2: Derive the Answer")
        self.reasoning_trace.append(f"Total counted bounce events: {bounce_count}")
        self.reasoning_trace.append(f"\\boxed{{{self.answer}}}")

def main():
    parser = argparse.ArgumentParser(description="Generate custom control videos for Exp 3, 4, 5, 8 using Manim")
    parser.add_argument("--experiment", type=str, required=True, choices=["3", "4", "5", "8"], help="Experiment index to generate")
    parser.add_argument("--count", type=int, default=4, help="Event count N")
    parser.add_argument("--span", type=float, default=8.0, help="Temporal span S in seconds")
    parser.add_argument("--position", type=str, default="mid", choices=["early", "mid", "late"], help="Position of the active span (Exp 4)")
    parser.add_argument("--regularity", type=str, default="periodic", choices=["periodic", "jittered", "irregular"], help="Timing regularity (Exp 5)")
    parser.add_argument("--duration", type=float, default=24.0, help="Total video duration in seconds")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--out-dir", type=str, default="videos/custom_exp", help="Output directory")
    args = parser.parse_args()
    
    out_dir = Path(args.out_dir)
    (out_dir / "questions").mkdir(parents=True, exist_ok=True)
    (out_dir / "solutions").mkdir(parents=True, exist_ok=True)
    (out_dir / "question_text").mkdir(parents=True, exist_ok=True)
    (out_dir / "reasoning_traces").mkdir(parents=True, exist_ok=True)
    
    # Resolve names based on experiment
    if args.experiment == "3":
        base_name = f"exp3_n{args.count}_s{args.span}_seed{args.seed}"
    elif args.experiment == "4":
        base_name = f"exp4_n{args.count}_s{args.span}_pos{args.position}_seed{args.seed}"
    elif args.experiment == "5":
        base_name = f"exp5_n{args.count}_s{args.span}_reg{args.regularity}_seed{args.seed}"
    else:
        # Standard replication (regular spacing over entire video)
        base_name = f"exp8_n{args.count}_seed{args.seed}"
        args.span = args.duration
        args.position = "early"
        args.regularity = "periodic"
        
    # Check if files already exist to skip rendering
    final_video_path = out_dir / "questions" / f"{base_name}.mp4"
    final_json_path = out_dir / "questions" / f"{base_name}.json"
    if final_video_path.exists() and final_json_path.exists():
        print(f"Skipping {base_name} (already exists).")
        return

    print(f"Rendering: {base_name} (Count={args.count}, Span={args.span}s, Pos={args.position}, Reg={args.regularity})")
    
    scene = CustomBouncingBall(
        count=args.count,
        span=args.span,
        position=args.position,
        regularity=args.regularity,
        duration=args.duration,
        seed=args.seed
    )
    scene.render()
    
    expected_output = Path("manim_output/videos/480p15/CustomBouncingBall.mp4")
    if expected_output.exists():
        final_filename = f"{base_name}.mp4"
        shutil.move(str(expected_output), str(out_dir / "questions" / final_filename))
        
        # Write files
        with open(out_dir / "solutions" / f"{base_name}.txt", "w") as f:
            f.write(str(scene.answer))
            
        with open(out_dir / "question_text" / f"{base_name}.txt", "w") as f:
            f.write(scene.question_text)
            
        with open(out_dir / "reasoning_traces" / f"{base_name}.txt", "w") as f:
            f.write("\n".join(scene.reasoning_trace))
            
        # Write metadata JSON
        meta = {
            "experiment": args.experiment,
            "count": args.count,
            "span": args.span,
            "position": args.position,
            "regularity": args.regularity,
            "duration": args.duration,
            "seed": args.seed,
            "bounce_times": scene.bounce_times
        }
        with open(out_dir / "questions" / f"{base_name}.json", "w") as f:
            json.dump(meta, f, indent=2)
            
        print(f"✓ Saved files successfully for {base_name}")
        
        # Clean up temporary Manim folder
        if os.path.exists("manim_output"):
            shutil.rmtree("manim_output")
    else:
        print(f"Error: Render did not produce expected file for {base_name}")

if __name__ == "__main__":
    main()
