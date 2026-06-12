import sys
import importlib

def main():
    """
    Main entry point for routing and running Qwen3-VL interpretability experiments.
    
    Usage:
        python run_experiments.py <exp1|exp2|exp3|exp4|exp5|1|2|3|4|5> [args...]
    """
    if len(sys.argv) < 2:
        print("Qwen3-VL Interpretability Suite Runner")
        print("=======================================")
        print("Usage:")
        print("  python run_experiments.py <experiment> [args...]\n")
        print("Available Experiments:")
        print("  1 | exp1 : Spatio-Temporal Attention Dispersion / Entropy")
        print("  2 | exp2 : Hidden Representation Cosine Similarity / PCA Trajectory")
        print("  3 | exp3 : State Probing (ON/OFF classifications)")
        print("  4 | exp4 : Preprocessing Overrides & Ablation Study")
        print("  5 | exp5 : Logit Lens Vocabulary Projection Profile")
        sys.exit(1)
        
    exp_key = sys.argv[1].lower()
    
    # Map input route to target module name in src
    route_map = {
        "1": "src.exp1_attention_dispersion",
        "exp1": "src.exp1_attention_dispersion",
        "2": "src.exp2_representation_similarity",
        "exp2": "src.exp2_representation_similarity",
        "3": "src.exp3_linear_probing",
        "exp3": "src.exp3_linear_probing",
        "4": "src.exp4_preprocessing_ablation",
        "exp4": "src.exp4_preprocessing_ablation",
        "5": "src.exp5_logit_lens",
        "exp5": "src.exp5_logit_lens",
    }
    
    if exp_key not in route_map:
        print(f"Error: Unknown experiment selection '{sys.argv[1]}'")
        print("Choose from: 1, 2, 3, 4, 5, exp1, exp2, exp3, exp4, exp5")
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
