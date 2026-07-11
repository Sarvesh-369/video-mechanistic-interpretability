import sys
import importlib
import re

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
        "3": "src.run_new_evaluations",
        "exp3": "src.run_new_evaluations",
        "4": "src.run_new_evaluations",
        "exp4": "src.run_new_evaluations",
        "5": "src.run_new_evaluations",
        "exp5": "src.run_new_evaluations",
        "6": "src.run_new_evaluations",
        "exp6": "src.run_new_evaluations",
        "7": "src.run_new_evaluations",
        "exp7": "src.run_new_evaluations",
        "8": "src.run_new_evaluations",
        "exp8": "src.run_new_evaluations",
    }
    
    if exp_key not in route_map:
        print(f"Error: Unknown experiment selection '{sys.argv[1]}'")
        print("Choose from: 1, 2, 3, 4, 5, 6, 7, 8, exp1, exp2, exp3, exp4, exp5, exp6, exp7, exp8")
        sys.exit(1)
        
    target_module = route_map[exp_key]
    
    # Extract the experiment number from exp_key
    match = re.match(r'exp(\d+)', exp_key)
    if not match:
        match = re.match(r'(\d+)', exp_key)
        
    # Slice sys.argv: keep the runner script name as argv[0],
    # discard the route name (argv[1]), and forward everything else.
    if target_module == "src.run_new_evaluations" and match:
        exp_num = match.group(1)
        sys.argv = [sys.argv[0], "--experiment", exp_num] + sys.argv[2:]
    else:
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
