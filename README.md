# Qwen3-VL Mechanistic Interpretability Guide: Probing the Temporal Interpolation Hypothesis

This guide details the theoretical and mathematical frameworks of the interpretability experiments designed to analyze the **Low-Frequency Trap** and test the **Temporal Interpolation Hypothesis** in Video-Language Models (VLMs) like Qwen3-VL.

---

## Theoretical Context & Reviewer Feedback

Large multimodal models fail catastrophically at counting conceptually simple, visually distinct, low-frequency events when the count $N$ exceeds a small threshold (e.g., $N > 4$). The **Temporal Interpolation Hypothesis** posits that models encode discrete temporal events as continuous appearance changes (temporal smoothing) rather than as sequence segment boundaries. 

The experiments in this suite target two primary architectural areas:
1.  **Decoder Attention Dynamics**: Investigating if attention dispersion over long contexts causes the model to lose track of events.
2.  **Representational Characteristics**: Testing if the visual features representation dynamically smoothly interpolates states across time, destroying the discrete event transitions.

---

## Detailed Experiment Overview (Theoretical Foundations)

### Experiment 1: Spatio-Temporal Attention Dispersion
*   **Target Mechanism**: Self-attention maps in unified transformer decoder layers.
*   **Core Question**: Does attention from the query token (answering the question) disperse over visual tokens as the temporal sequence gets longer?
*   **Theoretical Foundation**:
    We isolate the self-attention weights from the query token (which requests the count prediction) to the sequence of visual patches. Because visual tokens are mapped to temporal blocks, we aggregate spatial visual tokens to compute the temporal attention probability distribution:
    $$P_L^H(t) = \sum_{j \in \text{patches}(t)} A_{L,H}[\text{query}, j]$$
    where $A_{L,H}$ is the attention matrix at layer $L$ and head $H$. To quantify the dispersion of attention across the time steps $t \in [0, T-1]$, we compute the **Temporal Attention Entropy**:
    $$H_L^H(\text{temporal}) = -\sum_{t=0}^{T-1} P_L^H(t) \log (P_L^H(t) + \epsilon)$$
    
    *   **Max Possible Entropy**: $H_{\max} = \log T$ (corresponds to a uniform distribution).
*   **Theoretical Interpretation**: 
    *   If attention dispersion is the root cause, $H(\text{temporal})$ will rise sharply in later layers for videos with $N > 4$, indicating the query cannot focus on specific transition frames and instead "smears" attention across the entire video.
    *   If attention remains highly peaked, the bottleneck is representational rather than attention-routing.

---

### Experiment 2: Representation Similarity and State Space Trajectories
*   **Target Mechanism**: Cosine similarity and dimensionality reduction (PCA) on intermediate hidden states.
*   **Core Question**: Does the model represent video as a smooth continuous trajectory (interpolation) or as a sequence of discrete states?
*   **Theoretical Foundation**:
    For each temporal block $t \in [0, T-1]$, we take the spatial-mean pooled representation vector $h_t \in \mathbb{R}^{D}$ from transformer layer $L$. We evaluate the trajectory in state space:
    1.  **Consecutive Similarity**: $S(t) = \text{CosineSimilarity}(h_t, h_{t+1})$ measures the rate of representational change.
    2.  **Drift Relative to Origin**: $S_{\text{init}}(t) = \text{CosineSimilarity}(h_t, h_0)$ tracks displacement from the initial state.
    3.  **PCA Projection**: Projecting the high-dimensional vectors $h_t$ onto their first two principal components maps the state trajectory.
*   **Theoretical Interpretation**:
    *   **Discrete State Representation (Falsifying Interpolation)**: $S(t)$ remains near $1.0$ during steady states and drops sharply at transition frames (forming a step-like curve). The PCA plot shows distinct, well-separated clusters (e.g., one cluster for ON, one for OFF) with rapid jumps.
    *   **Continuous Temporal Interpolation (Supporting Interpolation)**: $S(t)$ remains flat and high, while $S_{\text{init}}(t)$ drifts slowly. The PCA plot reveals a smooth, continuous path connecting the start and end states, proving visual attributes are continuously smoothed.

---

### Experiment 3: Linear Probing for Perceptual State Preservation
*   **Target Mechanism**: Supervised classification probe on intermediate layer features.
*   **Core Question**: Is the frame-level state information (e.g., ON vs. OFF) still present in the hidden states of failing instances?
*   **Theoretical Foundation**:
    We train a linear classifier (the probe) to map the hidden state representation $h_t$ at a target layer to the ground-truth binary state label $y_t \in \{0, 1\}$ (ON vs. OFF) derived from the transition event logs.
    *   **Train Context**: Low-count successful videos ($N \le 3$), where the model count is correct.
    *   **Test Context**: High-count failing videos ($N \ge 5$), where the model count is incorrect.
*   **Theoretical Interpretation**:
    *   **Reasoning Bottleneck**: High probe accuracy ($>90\%$) on the failing target instances proves that the visual tokens *perceptually contain* the correct state transitions, but the transformer self-attention layers fail to count or aggregate them.
    *   **Perceptual Bottleneck**: Low probe accuracy on the failing instances indicates that the state information itself was lost or blended in the early layers, making accurate downstream reasoning impossible.

---

### Experiment 4: Preprocessing Ablation
*   **Target Mechanism**: Dynamic resolution pooling and frame subsampling thresholds.
*   **Core Question**: Does frame subsampling or dynamic downscaling destroy the transition logic?
*   **Theoretical Foundation**:
    We decouple visual perception limits from transformer layer context limits by systematically overriding preprocessing constraints:
    1.  **Temporal Resolution**: Forcing higher frame rate sampling (FPS).
    2.  **Spatial Resolution**: Forcing high spatial resolution bounds (disabling spatial patch pooling/resizing).
    3.  **Joint Resolution**: Applying both overrides.
*   **Theoretical Interpretation**:
    *   If high FPS / high resolution overrides shift the failure boundary (restoring correct counts), the root cause is information destruction during preprocessing.
    *   If accuracy remains low, the bottleneck resides inside the transformer decoder reasoning layers.

---

### Experiment 5: Logit Lens Analysis
*   **Target Mechanism**: Layer-wise hidden state projection onto vocabulary space.
*   **Core Question**: At what layer depth does the representation of the count collapse?
*   **Theoretical Foundation**:
    For the query token predicting the answer, we project the hidden state $h_L$ at every layer $L \in [0, \text{num-layers}]$ directly onto the vocabulary:
    $$\text{logits}_L = \text{lm-head}(\text{layer-norm}(h_L))$$
    $$\text{probs}_L = \text{Softmax}(\text{logits}_L)$$
    We track the probability of the correct count token (e.g., "5") vs. incorrect/under-counted tokens (e.g., "4") across the depth of the network.
*   **Theoretical Interpretation**:
    *   **Late-Stage Reasoning Suppression**: The correct count token has high probability in middle layers but drops in late layers, indicating the model computes the correct answer but overrides it during output generation.
    *   **Early-Stage Aggregation Failure**: The correct token never emerges, and the interpolated/under-counted tokens dominate across all layers.

---

## Instructions for Running on the Server

All scripts can be run in **Multi-Domain Mode** (defaulting to run on all 3 domains sequentially), **Directory Cohort Mode** (for a specific domain directory), or **Single-Video Mode** (for case study visualizations). Launch them via the root `run_experiments.py` wrapper.

### Setup Requirements
Ensure the environment contains the required packages:
```bash
pip install torch torchvision transformers accelerate qwen-vl-utils scikit-learn matplotlib num2words
```

### Output Directory Structure and Prompt Modes
All experiments (except Experiment 3) support the `--prompt-mode` CLI parameter with choices `cot` (Chain-of-Thought reasoning, default) and `direct` (Zero-Shot Direct Answer). Results for each experiment are automatically isolated in prompt-mode subdirectories under the domain-specific output directory:
- `results/expX/<domain_name>/cot/`
- `results/expX/<domain_name>/direct/`

---

### 1. Run Attention Dispersion Analysis (Exp 1)

*   **Multi-Domain Mode (Runs on all 3 domains sequentially)**:
    ```bash
    python run_experiments.py exp1 --model-id Qwen/Qwen3-VL-8B-Instruct --device cuda
    ```
*   **Directory Cohort Mode (Specific domain)**:
    ```bash
    python run_experiments.py exp1 --video-dir videos/temporal/bounce_ball --model-id Qwen/Qwen3-VL-8B-Instruct --device cuda
    ```
*   **Single-Video Mode (Case Study)**:
    ```bash
    python run_experiments.py exp1 --video-path videos/temporal/blinking/questions/sweep_count_blinks_c10_f0.5_s0_d24.0_count_blinks.mp4 --device cuda
    ```

---

### 2. Run Representation Trajectory Analysis (Exp 2)

*   **Multi-Domain Mode (Runs on all 3 domains sequentially)**:
    ```bash
    python run_experiments.py exp2 --model-id Qwen/Qwen3-VL-8B-Instruct --device cuda
    ```
*   **Directory Cohort Mode (Specific domain)**:
    ```bash
    python run_experiments.py exp2 --video-dir videos/temporal/bounce_ball --model-id Qwen/Qwen3-VL-8B-Instruct --device cuda
    ```
*   **Single-Video Mode (Case Study)**:
    ```bash
    python run_experiments.py exp2 --video-path videos/temporal/blinking/questions/sweep_count_blinks_c10_f0.5_s0_d24.0_count_blinks.mp4 --device cuda
    ```

---

### 3. Run State Probing (Exp 3)
*Performs linear state classification mapping targets (`OFF/ON` for Blinking, `Wall A/B` for Bounce, `Cool/Warm` colors for State Machine).*

*   **Multi-Domain Mode (Runs on all 3 domains sequentially)**:
    ```bash
    python run_experiments.py exp3 --max-train-videos 100 --regularization-c 0.1 --device cuda
    ```
*   **Directory Cohort Mode (Specific domain)**:
    ```bash
    python run_experiments.py exp3 --train-dir videos/temporal/bounce_ball --test-dir videos/temporal/bounce_ball --max-train-videos 100 --regularization-c 0.1 --device cuda
    ```
*   **Single-Video Mode (Case Study)**:
    ```bash
    python run_experiments.py exp3 --train-dir videos/temporal/blinking --video-path videos/temporal/blinking/questions/sweep_count_blinks_c10_f0.5_s0_d24.0_count_blinks.mp4 --device cuda
    ```

---

### 4. Run Preprocessing Ablation (Exp 4)

*   **Multi-Domain Mode (Runs on all 3 domains sequentially)**:
    ```bash
    python run_experiments.py exp4 --model-id Qwen/Qwen3-VL-8B-Instruct --device cuda
    ```
*   **Directory Cohort Mode (Specific domain)**:
    ```bash
    python run_experiments.py exp4 --video-dir videos/temporal/bounce_ball --device cuda
    ```
*   **Single-Video Mode (Case Study)**:
    ```bash
    python run_experiments.py exp4 --video-path videos/temporal/blinking/questions/sweep_count_blinks_c10_f0.5_s0_d24.0_count_blinks.mp4 --device cuda
    ```

---

### 5. Run Logit Lens (Exp 5)

*   **Multi-Domain Mode (Runs on first failing video in all 3 domains)**:
    ```bash
    python run_experiments.py exp5 --model-id Qwen/Qwen3-VL-8B-Instruct --device cuda
    ```
*   **Directory Cohort Mode (Runs on first failing video in specific domain)**:
    ```bash
    python run_experiments.py exp5 --video-dir videos/temporal/bounce_ball --device cuda
    ```
*   **Single-Video Mode (Case Study)**:
    ```bash
    python run_experiments.py exp5 --video-path videos/temporal/blinking/questions/sweep_count_blinks_c10_f0.5_s0_d24.0_count_blinks.mp4 --device cuda
    ```
