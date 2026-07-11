import sys
import importlib

def main():
    """
    Main entry point for routing and running Qwen3-VL interpretability experiments.
    
    Usage:
        python run_experiments.py <experiment> [args...]
    """
    if len(sys.argv) < 2:
        print("Qwen3-VL Interpretability Suite Runner")
        print("=======================================")
        print("Usage:")
        print("  python run_experiments.py <experiment> [args...]\n")
        print("Available Experiments:")
        print("  1 | exp1 : State Probing (ON/OFF classifications)")
        print("  2 | exp2 : Generated Token Visual Attention Rollout")
        print("  3 | exp3 : Count × Temporal Span Sweep")
        print("  4 | exp4 : Temporal Position Control")
        print("  5 | exp5 : Matched-Input-Length Oracle Control")
        print("  6 | exp6 : Symbolic Evidence Control")
        print("  7 | exp7 : Sequence Reconstruction Task")
        print("  8 | exp8 : Confidence-Based Capacity Boundary Estimation")
        sys.exit(1)
        
    exp_key = sys.argv[1].lower()
    
    # Map input route to target module name in src
    route_map = {
        "1": "src.exp1_linear_probing",
        "exp1": "src.exp1_linear_probing",
        "2": "src.exp2_generated_token_attention",
        "exp2": "src.exp2_generated_token_attention",
        "3": "src.exp3_count_span",
        "exp3": "src.exp3_count_span",
        "4": "src.exp4_temporal_position",
        "exp4": "src.exp4_temporal_position",
        "5": "src.exp5_matched_oracle",
        "exp5": "src.exp5_matched_oracle",
        "6": "src.exp6_symbolic_control",
        "exp6": "src.exp6_symbolic_control",
        "7": "src.exp7_sequence_reconstruction",
        "exp7": "src.exp7_sequence_reconstruction",
        "8": "src.exp8_boundary_estimation",
        "exp8": "src.exp8_boundary_estimation",
    }
    
    if exp_key not in route_map:
        print(f"Error: Unknown experiment selection '{sys.argv[1]}'")
        print("Choose from: 1, 2, 3, 4, 5, 6, 7, 8, exp1, exp2, exp3, exp4, exp5, exp6, exp7, exp8")
        sys.exit(1)
        
    target_module = route_map[exp_key]
    
    # Slice sys.argv: keep the runner script name as argv[0],
    # discard the route name (argv[1]), and forward everything else.
    sys.argv = [sys.argv[0]] + sys.argv[2:]
    
    print(f"[Runner] Launching {target_module}...")
    print(f"[Runner] Forwarded arguments: {sys.argv[1:]}\n")
    
    try:
        # Dynamically load the experiment module and execute its main entry point
        module = importlib.import_module(target_module)
        if hasattr(module, "main"):
            module.main()
        else:
            print(f"Error: Target module '{target_module}' does not expose a main() function.")
            sys.exit(1)
    except Exception as e:
        print(f"Runtime error occurred while executing {target_module}:")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
