# The Scientific Narrative & Story of Each Experiment

This document outlines the logical thread and scientific "story" behind each of the 5 experiments in our mechanistic interpretability suite. Together, these experiments diagnose and explain the **Low-Frequency Trap** in `Qwen3-VL-8B-Instruct`.

---

## The Core Scientific Question
> Why do state-of-the-art vision-language models (VLMs) fail to count simple repetitive events (like blinking lights or bouncing balls) in longer, slow-moving videos (the Low-Frequency Trap), even though they succeed on shorter videos?

We test the **Temporal Interpolation Hypothesis**: Does the model fail because its temporal attention gets dispersed/diluted (Hypothesis A), or because its internal representations of the states collapse or drift over time, failing to support a robust counting ledger (Hypothesis B)?

---

## Experiment 1: Spatio-Temporal Attention Dispersion
*   **The Question:** *Does the model fail in the Low-Frequency Trap because its attention gets "washed out" or dispersed across too many frames over time?* (Testing Hypothesis A).
*   **What We Do:** 
    *   Load the model in `eager` attention mode.
    *   Extract the self-attention weights from the query vector of the last prompt token (right before generation begins) back to the visual tokens of the video.
    *   Map these visual tokens back to their respective video frames ($t \in [0, T-1]$) and compute the Shannon entropy of the attention distribution across all layers and attention heads.
*   **The Comparison:** 
    *   **Cohort A (Easy, $N \le 3$, $f \le 1.0$ Hz):** Establishes the baseline entropy profile of a successful run.
    *   **Cohort B (Trap, $N \ge 5$, $f \le 1.0$ Hz):** Check if entropy spikes or flatlines (indicating attention dispersion).
*   **The Story:** 
    *   *If Cohort B attention entropy remains low and similar to Cohort A*, we prove that **attention dispersion is NOT the root cause**. The model is still looking at the correct frames at the correct times; the failure must lie deeper, in the representation space.
    *   This sets the stage for checking the internal representations in Experiment 2.

---

## Experiment 2: Representation Similarity & Trajectory Collapse
*   **The Question:** *Do the hidden state representations of events collapse, drift, or smooth out as the video duration increases?* (Testing Hypothesis B).
*   **What We Do:**
    *   Extract the sequence of visual token hidden states at a late transformer layer (e.g., Layer `-2`).
    *   Compute the frame-to-frame cosine similarity trajectory $S(t) = \text{CosineSimilarity}(h_t, h_{t+1})$ to check if event boundaries (state transitions) are represented as sharp, distinct jumps or get smoothed out.
    *   Apply PCA (Principal Component Analysis) to project the high-dimensional hidden trajectories into 2D space to visualize the model's "state space".
*   **The Comparison:**
    *   **Cohort A (Easy):** We expect a clear, periodic trajectory in PCA space (e.g., distinct loops or jumps corresponding to discrete state transitions).
    *   **Cohort B (Trap):** We expect the trajectory to spiral inward, flatten, or drift, indicating that the representation of later transitions collapses into a homogeneous cluster.
    *   **Cohort C (High-Freq, $N \ge 5$, $f \ge 3.0$ Hz):** Check if rapid transitions accelerate this representation collapse.
*   **The Story:**
    *   This experiment visualizes the physical collapse of the state representations. It proves that over long periods, the VLM's hidden states lose the ability to maintain distinct representations of the events.
    *   *But does this collapse mean the model has completely forgotten the physical state of the frame (e.g., did it not see the flash), or has it just failed to keep count?* We answer this in Experiment 3.

---

## Experiment 3: Linear Probing for Perceptual State Preservation
*   **The Question:** *Are the basic perceptual features (e.g., light is ON vs. OFF, ball is hitting a wall vs. floating) still encoded in the representations during a failure run, or does the model fail to perceive the events entirely?*
*   **What We Do:**
    *   Extract representations from Layer `-2` of the model.
    *   Train a linear classifier (Logistic Regression probe) on the representations of **Cohort A** (where the model successfully tracks state changes) using the ground-truth state labels (ON/OFF) from the video generation traces.
    *   Evaluate the trained probe on **Cohort B** (the Trap videos) where the model behaviorally fails to count.
*   **The Comparison:**
    *   **Training:** Cohort A representations (stable, healthy).
    *   **Evaluation:** Cohort B representations (failing behavior).
*   **The Story:**
    *   *If the linear probe successfully classifies the physical state (e.g., >90% F1-score) in Cohort B*, it proves that **perception is preserved**. The model's hidden states still contain the raw physical information of each event. 
    *   The failure is therefore a **bookkeeping/aggregation failure** (the model sees the flash, but cannot increment or maintain the counter), rather than a visual blindness issue.

---

## Experiment 4: Preprocessing Ablation & Boundary Rescue
*   **The Question:** *Can we behaviorally "rescue" the model from the Low-Frequency Trap by overriding the visual preprocessing configuration (FPS and Resolution)?*
*   **What We Do:**
    *   Select videos right on the failure boundary ($4 \le N \le 6$, $f \le 1.0$ Hz).
    *   Evaluate the model's counting accuracy under four configurations:
        1.  *Baseline:* Normal processor config.
        2.  *High Temporal:* Force higher frame rate (`fps=4.0`).
        3.  *High Spatial:* Force maximum spatial resolution (`max_pixels=602112`).
        4.  *High Temporal + Spatial:* Apply both overrides.
*   **The Story:**
    *   This experiment bridges the internal representations back to external model behavior. 
    *   By showing that increasing temporal resolution (FPS) or spatial resolution rescues the model's accuracy on boundary videos, we demonstrate that the representational collapse is sensitive to the token-density of the input, pointing to practical engineering mitigations for VLMs.

---

## Experiment 5: Logit Lens
*   **The Question:** *Where in the network's layers does the counting representation break down?*
*   **What We Do:**
    *   For the final query token predicting the answer, extract the intermediate representation $h_L$ at every layer $L \in [0, \text{num\_layers}-1]$.
    *   Apply Layer Normalization and the Language Model Head (`lm_head`) to project $h_L$ directly to the vocabulary probability space.
    *   Track the probability trajectory of the correct count token (e.g., "5") vs. under-counted tokens (e.g., "4", "3") across the depth of the network.
*   **The Comparison:**
    *   **Cohort A (Success case):** The correct count token should rise steadily to dominance in the middle-to-late layers.
    *   **Cohort B (Trap case):** The model may start with the correct count in early/middle layers, but late layers project to under-counted vocabulary tokens (or vice versa), locating the exact layer depth of representation corruption.
*   **The Story:**
    *   The Logit Lens is the final chapter of our story. It acts as an X-ray, pinpointing the exact layer depth where the model's internal bookkeeping ledger is corrupted and replaced by an under-counted answer.
