# Experimental Methodology & Setup: Qwen3-VL Temporal Reasoning

This document details the mathematical formulations, step-by-step algorithms, tensor shapes, and concrete walkthrough examples for each of the 5 experiments in our mechanistic interpretability suite. These experiments investigate the **Low-Frequency Trap** and test the **Temporal Interpolation Hypothesis** in `Qwen3-VL-8B-Instruct`.

---

## 1. Unified Dataset Design & Video Cohort Strategy

To analyze the internal mechanics of the VLM, we isolate the two variables: event count $N$ and event frequency $f$. We design three controlled cohorts:

*   **Cohort A: Easy / Successful Baseline (Control)**
    *   **Criteria:** Event count $N \le 3$, event frequency $f \le 1.0$ Hz.
    *   **Scientific Role:** Establishes the reference state of the model when it successfully tracks and counts events.
*   **Cohort B: Hard / Trap (The Low-Frequency Trap)**
    *   **Criteria:** Event count $N \ge 5$, event frequency $f \le 1.0$ Hz.
    *   **Scientific Role:** Captures the failure state where the model fails to count despite slow, visually distinct transitions.
*   **Cohort C: High-Frequency / Crowded (Contrastive Baseline)**
    *   **Criteria:** Event count $N \ge 5$, event frequency $f \ge 3.0$ Hz.
    *   **Scientific Role:** Tests if rapid dynamics accelerate representational collapse or attention dispersion.

---

## 2. Deep Experimental Specifications

### Experiment 1: Spatio-Temporal Attention Dispersion

#### 1. Mathematical Formulation
Let the input token sequence be $x = [x_0, x_1, \dots, x_{S-1}]$, where $S$ is the total sequence length. The self-attention matrix at layer $L$ and attention head $H$ is computed as:
$$A_{L,H} = \text{Softmax}\left(\frac{Q_{L,H} K_{L,H}^T}{\sqrt{d_k}}\right) \in [0, 1]^{S \times S}$$
where $Q_{L,H}, K_{L,H} \in \mathbb{R}^{S \times d_k}$ are the query and key projection matrices. 

We isolate the attention weights from the query vector of the prompt-end token (at index $q = S - 1$, right before generation begins) back to the visual tokens representing the video frames. Let these visual tokens span a contiguous range of indices $v \in [I_{\text{start}}, I_{\text{end}} - 1]$. The visual attention vector is defined as:
$$a_{L,H} = A_{L,H}[q, I_{\text{start}}:I_{\text{end}}] \in [0, 1]^{V}$$
where $V = I_{\text{end}} - I_{\text{start}}$ is the total number of visual tokens.

Using the spatial-temporal grid dimensions $(T, H, W)$ of the video, we group the $V$ visual tokens into $T$ frames (each frame consisting of $H \times W$ spatial patches). The temporal attention probability distribution $P_L^H(t)$ for $t \in [0, T-1]$ is computed by summing the spatial attention weights:
$$P_L^H(t) = \sum_{j = t \cdot H \cdot W}^{(t+1) \cdot H \cdot W - 1} a_{L,H}[j]$$
If the sum of attention weights to visual tokens is zero (due to numerical underflow), we apply a uniform fallback distribution:
$$P_L^H(t) = \frac{1}{T}, \quad \forall t \in [0, T-1]$$
The Shannon entropy of this temporal attention distribution is:
$$H_L^H(\text{temporal}) = -\sum_{t=0}^{T-1} P_L^H(t) \log\left(P_L^H(t) + \epsilon\right)$$
where $\epsilon = 10^{-9}$ is a smoothing factor to prevent $\log(0)$ errors.

#### 2. Step-by-Step Algorithm & Tensor Shapes
1.  **Tokenize Prompt and Video**: Tokenize the inputs, yielding `input_ids` of shape `(1, S)`.
2.  **Register Attention Hooks**: Register a forward hook on the self-attention submodule of each transformer block.
3.  **Forward Pass**: Execute a forward pass with the model.
4.  **On-the-Fly Hook Slicing**: During the forward pass, each hook intercepts the attention weights tensor `attn_weights` of shape `(1, num_heads, S, S)`.
    *   Slice the tensor to extract the row for the query token at position `-1` across the visual token indices: `attn_weights[:, :, -1, start_idx:end_idx]`, yielding a tensor of shape `(1, num_heads, V)`.
    *   Clone the slice and move it to CPU memory.
    *   Reassign `attn_weights` to a dummy tensor of shape `(1, 1, 1, 1)` to allow PyTorch to immediately free the massive `(S, S)` activation memory.
5.  **Temporal Pooling**: Group the `V` visual tokens into `T` chunks of size `H * W`. Sum the attention weights across each chunk, resulting in a temporal attention weight tensor of shape `(num_heads, T)`.
6.  **Calculate Entropy**: Normalize the temporal attention weights to sum to $1.0$, and calculate the Shannon entropy for each head and layer.

#### 3. Concrete Example
*   **Video:** Blinking task domain, file name `sweep_count_blinks_c2_f0.5_s9_d24.0_count_blinks.mp4` (Cohort A, Ground Truth = 2, duration = 24 seconds, FPS = 1.0).
*   **Grid Dimensions:** $T=24$ frames, $H=14$ spatial patches, $W=14$ spatial patches.
*   **Visual Tokens:** $V = 24 \times 14 \times 14 = 4704$ tokens.
*   **Text Prompt:** "How many times did the object flash?" (tokenized to 250 tokens).
*   **Total Sequence Length:** $S = 250 + 4704 = 4954$ tokens.
*   **Tensors:**
    *   `input_ids` shape: `(1, 4954)`.
    *   Raw `attn_weights` shape: `(1, 40, 4954, 4954)`.
    *   Sliced attention shape: `(1, 40, 4704)`.
    *   Aggregated temporal weights shape: `(40, 24)`.
    *   Normalized distribution $P_L^H(t)$ shape: `(40, 24)`.
    *   Maximum possible entropy: $\log(24) \approx 3.178$ nats.

---

### Experiment 2: Representation Similarity and Space Trajectories

#### 1. Mathematical Formulation
Let the hidden state representation at layer $L$ be $H_L \in \mathbb{R}^{S \times D}$, where $D$ is the model's hidden dimension. We extract the visual token representations $h^{\text{visual}} \in \mathbb{R}^{V \times D}$ and apply spatial mean pooling to obtain the temporal state sequence:
$$\bar{h}_t = \frac{1}{H \cdot W} \sum_{j = t \cdot H \cdot W}^{(t+1) \cdot H \cdot W - 1} h^{\text{visual}}_j \in \mathbb{R}^{D}, \quad t \in [0, T-1]$$
To analyze state changes without the static background spatial bias and high-dimensional anisotropy (the "cone effect" where all token representations cluster closely together), we subtract the temporal mean vector:
$$\mu = \frac{1}{T} \sum_{t=0}^{T-1} \bar{h}_t \in \mathbb{R}^{D}$$
$$\tilde{h}_t = \bar{h}_t - \mu \in \mathbb{R}^{D}$$
We then calculate three metrics:
1.  **Consecutive Similarity (Pearson Correlation)**:
    $$C(t) = \frac{\langle \tilde{h}_t, \tilde{h}_{t+1} \rangle}{\|\tilde{h}_t\|_2 \|\tilde{h}_{t+1}\|_2} \in [-1, 1], \quad t \in [0, T-2]$$
2.  **Initial Similarity**:
    $$C_{\text{init}}(t) = \frac{\langle \tilde{h}_t, \tilde{h}_0 \rangle}{\|\tilde{h}_t\|_2 \|\tilde{h}_0\|_2} \in [-1, 1], \quad t \in [0, T-1]$$
3.  **PCA Projection**: We apply Principal Component Analysis (PCA) to project the centered trajectory matrix $\tilde{H} = [\tilde{h}_0, \tilde{h}_1, \dots, \tilde{h}_{T-1}]^T \in \mathbb{R}^{T \times D}$ into a 2D space:
    $$z_t = \tilde{h}_t \cdot W_{\text{PCA}} \in \mathbb{R}^{2}$$
    where $W_{\text{PCA}} \in \mathbb{R}^{D \times 2}$ is the matrix containing the top two principal components of $\tilde{H}$.

#### 2. Step-by-Step Algorithm & Tensor Shapes
1.  **Extract Hidden States**: Perform a forward pass and extract the hidden states tensor at layer `-2`, yielding `hidden_states` of shape `(1, S, D)`.
2.  **Extract Visual Sequence**: Slice the contiguous visual token region to obtain a tensor of shape `(V, D)`.
3.  **Reshape to Spatial-Temporal Grid**: Reshape the visual tensor to shape `(T_out, H_out * W_out, D)` (accounting for PatchMerger convolution downsampling where $T_{\text{out}} = T // 2$, $H_{\text{out}} = H // 2$, and $W_{\text{out}} = W // 2$).
4.  **Spatial Mean Pooling**: Compute the mean along dimension $1$, resulting in the pooled trajectory `temporal_trajectory` of shape `(T_out, D)`.
5.  **Mean-Centering**: Convert values to `float32`. Compute the mean vector `mu` of shape `(D,)` and subtract it to get the centered trajectory matrix of shape `(T_out, D)`.
6.  **Compute Similarities**: Compute the cosine similarities between adjacent steps (producing `T_out - 1` values) and between each step and the initial step (producing `T_out` values).
7.  **PCA Projection**: Fit and project the trajectory using a 2-component PCA solver, yielding coordinates of shape `(T_out, 2)`.

#### 3. Concrete Example
*   **Video:** Bounce Ball task domain, file name `cohort_A_sweep_count_bounces_c2_f0.5_s9_d24.0_count_bounces_repr.png` (Cohort A, Ground Truth = 2, duration = 24 seconds, FPS = 2.0).
*   **Hidden Dimension:** $D = 4096$.
*   **Frames:** 48 frames at 2.0 FPS.
*   **Pooled Steps:** $T_{\text{out}} = 48 // 2 = 24$ steps.
*   **Tensors:**
    *   `hidden_states` shape: `(1, 4954, 4096)`.
    *   `temporal_trajectory` shape: `(24, 4096)`.
    *   Mean-centered correlation array length: $23$ values.
    *   PCA projection coordinates shape: `(24, 2)`.

---

### Experiment 3: Linear Probing for Perceptual State Preservation

#### 1. Mathematical Formulation
Let the training dataset consist of spatial-mean pooled hidden states extracted from healthy baseline instances (Cohort A). For each frame representation $h_i \in \mathbb{R}^{D}$, we pair it with a ground-truth binary state label $y_i \in \{0, 1\}$ (e.g. `OFF` vs `ON`).

We train a Logistic Regression probe by minimizing the L2-regularized negative log-likelihood:
$$\min_{w, b} \frac{1}{2} w^T w + C_{\text{reg}} \sum_{i=1}^{M} \log\left(1 + \exp\left(-y_i (w^T h_i + b)\right)\right)$$
where $C_{\text{reg}} = 0.1$ is the inverse regularization strength, and $M$ is the number of cropped training frames. To handle class imbalance, we scale the loss by class weights:
$$W_c = \frac{M}{2 \cdot M_c}$$
where $M_c$ is the number of samples in class $c$.

During evaluation, the frozen probe projects representations from failing Cohort B runs to predict the binary states:
$$\hat{y}_t = \mathbb{I}\left[\sigma(w^T h_t + b) \ge 0.5\right]$$
Active temporal cropping is applied to discard inactive tail frames:
$$\text{Crop Condition: } t \cdot \left(\frac{\text{duration}}{T}\right) \le \text{last\_event\_time} + \text{crop\_buffer}$$
where $\text{crop\_buffer} = 1.0$s.

#### 2. Step-by-Step Algorithm & Tensor Shapes
1.  **Extract Training Features**: Extract visual representations from $N_{\text{train}}$ train videos (Cohort A).
2.  **Align Labels**: Parse reasoning traces to generate a binary label sequence matching the temporal steps.
3.  **Temporal Cropping**: Apply the crop condition to discard frames beyond the active window.
4.  **Concatenate Datasets**: Concatenate train features to `X_train` of shape `(M_train, D)` and labels to `y_train` of shape `(M_train,)`.
5.  **Train Probe**: Fit the Logistic Regression classifier on `X_train` and `y_train`.
6.  **Extract Test Features**: Extract representations from $N_{\text{test}}$ test videos (Cohort B), crop, and concatenate to `X_eval` of shape `(M_eval, D)` and `y_eval` of shape `(M_eval,)`.
7.  **Generate Predictions**: Predict states `y_eval_pred` of shape `(M_eval,)` and compute accuracy/F1-scores.

#### 3. Concrete Example
*   **Domain:** Blinking domain, training on 100 easy blinking videos.
*   **State Space:** Class `0` is `OFF` (flash event), Class `1` is `ON` (background).
*   **Trajectory Slicing:** A 24-second video has 24 steps at 1 FPS. If the last flash event occurs at 4.2 seconds, the cropping limit is $4.2 + 1.0 = 5.2$ seconds, keeping the first 6 frames and throwing away the remaining 18 static frames.
*   **Tensors:**
    *   `X_train` shape: `(1200, 4096)` (assuming 12 active frames per video across 100 videos).
    *   `y_train` shape: `(1200,)`.
    *   `X_eval` shape: `(180, 4096)` (assuming 12 active frames per video across 15 Cohort B test videos).
    *   `y_eval` shape: `(180,)`.
    *   `y_eval_pred` shape: `(180,)`.

---

### Experiment 4: Preprocessing Ablation

#### 1. Mathematical Formulation
Let the VLM's final parsed answer prediction be modeled as:
$$\text{Count} = \mathcal{M}\left(\text{Preprocessor}(V, \text{config})\right)$$
where $V$ is the raw video file and $\text{config}$ contains FPS and pixel resolution bounds.

We ablate four preprocessing configurations:
1.  **Baseline**: Processor default parameters (FPS = 1.0, $\text{max\_pixels} = 229376$).
2.  **High Temporal Resolution**: Force frame sampling to `4.0` FPS.
3.  **High Spatial Resolution**: Force image size to $\text{max\_pixels} = 602112$.
4.  **High Temporal & Spatial**: Combine both overrides (FPS = 4.0, $\text{max\_pixels} = 602112$).

Accuracy is measured strictly based on exact matches of LaTeX boxed parses:
$$\text{Accuracy} = \frac{1}{|K|} \sum_{k \in K} \mathbb{I}\left[\text{Parse}(\text{Response}_k) == \text{GT}_k\right]$$

#### 2. Step-by-Step Algorithm & Tensor Shapes
1.  **Filter Dataset**: Select videos at the trap boundary ($4 \le N \le 6$ events, frequency $f \le 1.0$ Hz).
2.  **Preprocess Video**: For each video, apply the selected configuration override in `process_vision_info`, generating `pixel_values_videos` of shape `(num_patches, channel * temporal_patch * spatial_patch)` and `video_grid_thw` of shape `(1, 3)`.
3.  **Forward Pass**: Run a forward pass to generate the response text.
4.  **Parse Boxed Count**: Parse the generated text using the regex pattern `\\boxed{(\\d+)}`.
5.  **Compute Metrics**: Compare the parsed count against the ground truth.

#### 3. Concrete Example
*   **Video:** State Machine domain, $N = 5$ transitions, duration = 24 seconds.
*   **Configuration:** High Temporal Resolution (FPS = 4.0).
*   **Tensors:**
    *   `video_grid_thw` shape: `[96, 14, 14]` (representing 96 temporal steps after 3D convolution patch pooling, and $14 \times 14 = 196$ spatial patches per step).
    *   `pixel_values_videos` shape: `(18816, 1176)` (representing 18816 total patches across the video sequence, each patch consisting of $2 \times 14 \times 14 \times 3$ raw pixel channels).

---

### Experiment 5: Logit Lens

#### 1. Mathematical Formulation
Let the hidden state representation of the final query token $q$ (the token predicting the next text character) at layer $L$ be $h_L \in \mathbb{R}^{D}$. 

We project this hidden state directly onto the vocabulary space using the model's final layer norm $LN$ and Language Model head $W_{LM} \in \mathbb{R}^{D \times \text{vocab\_size}}$:
$$\text{logits}_L = LN(h_L) \cdot W_{LM} \in \mathbb{R}^{\text{vocab\_size}}$$
$$\text{probs}_L = \text{Softmax}(\text{logits}_L) \in [0, 1]^{\text{vocab\_size}}$$
We track the probability trajectory of the correct digit token $d_{\text{GT}}$ and alternative digit tokens $d_{\text{alt}}$:
$$P_L(\text{correct}) = \text{probs}_L[d_{\text{GT}}]$$
$$P_L(\text{alt}) = \text{probs}_L[d_{\text{alt}}]$$

#### 2. Step-by-Step Algorithm & Tensor Shapes
1.  **Enable Hidden State Outputs**: Execute a forward pass with the model configuration parameter `output_hidden_states=True`.
2.  **Retrieve Layer Representations**: Extract the hidden states tuple of length `num_layers + 1`. Each tensor in the tuple has shape `(1, seq_len, D)`.
3.  **Iterate Layers**: For each layer $L \in [0, 36]$:
    *   Slice the final query token representation: `h_L = hidden_states[L][0, -1]` of shape `(D,)`.
    *   Apply the model's final layer normalization block: `normed_h = model.model.norm(h_L.unsqueeze(0))`, resulting in a tensor of shape `(1, D)`.
    *   Project to vocabulary space: `logits = model.lm_head(normed_h)[0]`, yielding a tensor of shape `(vocab_size,)`.
    *   Cast values to `float32`. Compute the softmax probability distribution `probs` of shape `(vocab_size,)`.
    *   Retrieve the probabilities corresponding to the correct digit token and target alternative digit tokens.

#### 3. Concrete Example
*   **Video:** Blinking domain failing video (Ground Truth = 5, duration = 24 seconds).
*   **Target Vocab Tokens:** `"5"` (correct count token), `"3"` (under-counted alternative attractor).
*   **Hidden Dimension:** $D = 4096$.
*   **Vocabulary Size:** $151646$ tokens.
*   **Tensors:**
    *   `h_L` shape: `(4096,)`.
    *   `normed_h` shape: `(1, 4096)`.
    *   `logits` shape: `(151646,)`.
    *   `probs` shape: `(151646,)`.
    *   Correct count token `"5"` ID: `20`.
    *   Alternative count token `"3"` ID: `18`.
