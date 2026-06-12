# Experimental Methodology & Setup: Qwen3-VL Temporal Reasoning

This document details the practical methods, configurations, and cohort designs utilized to investigate the **Low-Frequency Trap** and test the **Temporal Interpolation Hypothesis** in `Qwen3-VL-8B-Instruct`.

---

## 1. Unified Dataset Design & Video Cohort Strategy

To answer the scientific question of *why* the model fails, we must isolate distinct variables (namely, event count $N$ vs. event frequency $f$). Running complex internal representation extraction on the entire grid of 800+ videos is computationally expensive and makes visual comparisons difficult.

Instead, we employ a hybrid strategy:

1.  **Full Grid Sweep (Behavioral Mapping):** Applied to **Experiment 4** (Preprocessing Ablation) to map the shift in the model's external success/failure boundary across all frequencies and counts.
2.  **Targeted Video Cohorts (Internal Probing):** Applied to **Experiments 1, 2, 3, and 5** to compare healthy vs. degraded model states.

### The Diagnostic Cohorts

We divide our video datasets (`blinking`, `bounce_ball`, and `state_machine`) into three controlled cohorts:

*   **Cohort A: Easy / Successful Baseline (Control)**
    *   **Criteria:** Event count $N \le 3$, event frequency $f \le 1.0$ Hz.
    *   **Scientific Role:** Establishes the reference state of the model when it successfully tracks and counts events.
*   **Cohort B: Hard / Trap (The Low-Frequency Trap)**
    *   **Criteria:** Event count $N \ge 5$, event frequency $f \le 1.0$ Hz.
    *   **Scientific Role:** Captures the failure state where the model fails to count despite slow, visually distinct transitions. Tests whether representations collapse over time.
*   **Cohort C: High-Frequency / Crowded (Contrastive Baseline)**
    *   **Criteria:** Event count $N \ge 5$, event frequency $f \ge 3.0$ Hz.
    *   **Scientific Role:** Tests the *Temporal Interpolation Hypothesis* by checking if fast dynamics accelerate the blending of representations or attention dispersion.

---

## 2. Experimental Procedures (Step-by-Step)

### Experiment 1: Spatio-Temporal Attention Dispersion
*   **Methodology:**
    1. Load the model in **`eager` attention mode** to enable attention weights extraction (bypassing SDPA limitations).
    2. Input 5 random videos from **Cohort A** (Easy) and 5 from **Cohort B** (Trap).
    3. Retrieve self-attention weights from the query vector of the count-answer token back to all visual token key vectors.
    4. Group visual tokens by frame index $t$ using the layout parameters in `video_grid_thw`.
    5. Aggregate the weights spatially per frame to compute the temporal attention probability distribution $P_L^H(t)$.
    6. Compute the Shannon entropy $H_L^H(\text{temporal}) = -\sum_{t=0}^{T-1} P_L^H(t) \log P_L^H(t)$ across layers and heads.
*   **Key Parameters:**
    *   `model_id`: `Qwen/Qwen3-VL-8B-Instruct`
    *   `device`: `cuda`
    *   `attn_implementation`: `eager`
    *   `query_token_pos`: `-1` (last input token prior to generation)

---

### Experiment 2: Representation Similarity and Space Trajectories
*   **Methodology:**
    1. Extract visual representation trajectories from the middle-to-late layers of the transformer (e.g. layer `-2`).
    2. Contrast three representative test cases:
        *   **Cohort A case:** Low-count, low-frequency (e.g. $N=2, f=0.5$ Hz).
        *   **Cohort B case:** High-count, low-frequency (e.g. $N=6, f=0.5$ Hz).
        *   **Cohort C case:** High-count, high-frequency (e.g. $N=6, f=3.5$ Hz).
    3. Compute frame-to-frame cosine similarities $S(t) = \text{CosineSimilarity}(h_t, h_{t+1})$ to identify if transition boundaries are discrete (step-like drops) or smoothed.
    4. Apply Principal Component Analysis (PCA) to project the temporal trajectory vectors $h_t \in \mathbb{R}^D$ onto 2D space.
*   **Key Parameters:**
    *   `layer_idx`: `-2` (last representations layer prior to prediction)
    *   `torch_dtype`: `torch.bfloat16` (converted to `float32` before PCA/Similarity computations to avoid NumPy exceptions).

---

### Experiment 3: Linear Probing for Perceptual State Preservation
*   **Methodology:**
    1. **Train Phase:** Extract representations at Layer `-2` for 100 videos from **Cohort A** ($N \le 3$, $f \le 1.0$ Hz) where the model is highly accurate.
    2. Fit a Logistic Regression probe to classify the binary state of the frame (ON vs. OFF, Wall A vs. B, etc.) using trace logs as ground truth.
    3. **Generalization Test:** Evaluate this trained probe on 15 videos from **Cohort B** ($N \ge 5$, $f \le 1.0$ Hz).
    4. **Active Temporal Cropping:** Crop trajectories and labels to the active video segment: `time_step <= last_event_time + crop_buffer` (default `1.0`s buffer). This discards the long tail of static frames, balancing the training and evaluation support.
*   **Key Parameters:**
    *   `max_train_videos`: `100`
    *   `regularization_c`: `0.1` (inverse L2 regularization strength)
    *   `class_weight`: `balanced` (to handle minor remaining imbalances in the active window)
    *   `crop_active_only`: `True`

---

### Experiment 4: Preprocessing Ablation (Boundary Rescue)
*   **Methodology:**
    1. Select videos at the transition boundary of the Low-Frequency Trap: $4 \le N \le 6$ and $f \le 1.0$ Hz.
    2. Evaluate exact-match accuracy of the model's counting performance under four preprocessing overrides:
        *   **Baseline:** Normal processor parameters.
        *   **High Temporal:** Force frame rate to `4.0` FPS.
        *   **High Spatial:** Set `max_pixels = 602112` (the model's upper limit) to preserve spatial details.
        *   **High Temporal and Spatial:** Apply both overrides.
    3. Parse the output count exclusively from LaTeX `\boxed{}` formats (having set `max_new_tokens=1024` to ensure complete reasoning generation).
*   **Key Parameters:**
    *   `max_new_tokens`: `1024`
    *   `target_resolution`: `602112` pixels maximum

---

### Experiment 5: Logit Lens Analysis
*   **Methodology:**
    1. Select a failing video from **Cohort B** ($N \ge 5, f \le 1.0$ Hz) and a successful video from **Cohort A** ($N \le 3, f \le 1.0$ Hz).
    2. For the final query token predicting the answer, extract the intermediate representation $h_L$ at every layer $L$.
    3. Apply Layer Normalization (`norm`) and LM Head Projection (`lm_head`) to map $h_L$ directly to the vocabulary.
    4. Plot the layer-wise probabilities of the correct count token (e.g. "5") vs. under-counted tokens (e.g. "4", "3") to locate where in the network's depth the counting bookkeeping representation breaks down.
*   **Key Parameters:**
    *   `output_hidden_states`: `True`
    *   `torch_dtype`: `torch.bfloat16` (cast to `float32` prior to softmax and numpy conversions to prevent runtime exceptions).
