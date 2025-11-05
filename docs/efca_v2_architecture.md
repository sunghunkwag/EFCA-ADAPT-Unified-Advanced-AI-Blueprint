# EFCA-v2 Redesign: Detailed Structure and Integration Specification

**Objective:** Redesign a detailed architecture to implement measurable functional self-awareness (self-uncertainty modeling, internal parameter adjustment, self-directed goal generation) integrated within a single agent and enables end-to-end learning.

**Core Philosophy:** Based on predictive processing theory with hierarchical free energy minimization, securing learning stability and adaptability through metacognitive control, and inducing self-directed behavior through intrinsic motivation.

---

## 1. System Overview and Information Flow

EFCA-v2 has the following core modules that interact organically:

1. **Environment:** External world or simulation. Provides state `s_t`, receives action `a_t`, and returns next state `s_{t+1}` and reward `r_t^{ext}`.
2. **Sensor Preprocessing:** Receives raw sensor input `x_raw` (e.g., images, sounds) and converts it to the first layer input format `x_0` for H-JEPA (resizing, normalization, etc.).
3. **Hierarchical JEPA (H-JEPA):** Multi-layer structure that receives sensor input to generate abstract representations `z_k` (k=1...K), and calculates prediction errors `ΔE_k` by predicting the representations of the next state at each level. A core element of free energy minimization.
4. **Tokenizer & Sparse Global Workspace (s-GWT):** Tokenizes high-level representations (`z_K`) from H-JEPA and other salient information (e.g., large prediction errors), routing them to sparsely selected slots to create the workspace state `S_gwt` that corresponds to the current integrated "contents of consciousness."
5. **Continuous-Time Liquid Neural Network (CT-LNN):** Receives hierarchical representations `{z_k}` from H-JEPA, GWT state `S_gwt`, previous action `a_{t-1}` to model the continuous dynamic state `h(t)` of the system. The basis for short-term prediction and action decisions.
6. **Task Policy Network (Task Policy `π_task`):** Determines action `a_t` to interact with the current environment based on the CT-LNN state `h(t)` and GWT state `S_gwt`. Trained to maximize external reward `r_t^{ext}`.
7. **Bipartite Metacognition:**
   * **Probe Network:** Monitors internal system states (LNN state `h(t)`, JEPA errors `ΔE_k`, current meta parameters, etc.) to output metacognitive assessments `φ` (uncertainty `φ_unc`, competence `φ_comp`, effort `φ_eff`).
   * **Meta-Controller (Meta-Controller `π_meta`):** Receives the probe's output `φ` to determine meta-actions `a_meta` that adjust internal system parameters (JEPA layer weights `λ_k`, learning rate `α`, exploration rate `ε_explore`, GWT sparsity, etc.). Trained to maximize intrinsic motivation rewards.
8. **Dual-Axis Intrinsic Motivation:** Defines reward `r_meta` for meta-controller learning. Based on two axes: knowledge seeking (uncertainty reduction) and competence improvement (task performance improvement or progress toward internal goal achievement).
9. **Memory Mesh:** Compresses important experiences (high error, high reward, meta-controller intervention points, etc.) using VQ-VAE for long-term storage, and retrieval/use when needed (e.g., replay buffer, self-modeling).

**Information Flow Diagram (Conceptual):**

```mermaid
graph TD
    subgraph Environment
        EnvState[State s_t]
        EnvAction[Action a_t]
        EnvNextState[Next State s_{t+1}]
        EnvReward[External Reward r_t^{ext}]
    end

    subgraph Agent
        Sensor[Sensor Preprocessing]
        HJEPA[Hierarchical JEPA (K layers)]
        Tokenizer[Tokenizer]
        GWT[Sparse Global Workspace (s-GWT)]
        CTLNN[Continuous-Time LNN]
        TaskPolicy[Task Policy π_task]
        Probe[Probe Network]
        MetaControl[Meta-Controller π_meta]
        Motivation[Dual-Axis Intrinsic Motivation]
        Memory[Memory Mesh (VQ-VAE)]

        EnvState -- x_raw --> Sensor -- x_0 --> HJEPA
        HJEPA -- {z_k}, {ΔE_k} --> CTLNN
        HJEPA -- z_K, ΔE_k? --> Tokenizer -- tokens --> GWT
        GWT -- S_gwt --> CTLNN
        GWT -- S_gwt --> TaskPolicy
        CTLNN -- h(t) --> TaskPolicy
        CTLNN -- h(t) --> Probe
        TaskPolicy -- a_t --> EnvAction
        EnvAction -- a_{t-1} --> CTLNN

        HJEPA -- {ΔE_k} --> Probe
        MetaControl -- Current Params --> Probe
        Probe -- φ (unc, comp, eff) --> MetaControl
        MetaControl -- a_meta (λ_k, ε_explore, α...) --> HJEPA & TaskPolicy & Optimizer & GWT & CTLNN
        Motivation -- r_meta --> MetaControl

        Motivation -- Uses φ_unc, φ_comp, r_t^{ext}? --> Motivation

        Memory -- Store Significant Events --> Memory
        Memory -- Retrieve for Replay/Context --> HJEPA & CTLNN & TaskPolicy?

        EnvReward --> TaskPolicy  // For Task RL update
        EnvReward -- Maybe used by --> Motivation
        HJEPA -- Prediction Errors --> Motivation // For Epistemic Reward
    end

    EnvAction --> EnvNextState
    EnvAction --> EnvReward
```

(Note: Mermaid diagrams are visually rendered in markdown viewers/editors that support them.)

## 2. Mathematical Redefinition and Detailed Development

### 2.1 Hierarchical Free Energy (H-JEPA Focus)

Each JEPA layer k (from 1 to K) tries to minimize the following free energy term F_k:

$$
F_k(t) = E_{q(z_k|c_k)}[
\underbrace{D_{pred}(z_k, \hat{z}_k)}_\text{Energy: prediction accuracy}
+ \beta_k
\underbrace{KL[q(z_k|c_k) \parallel p(z_k)]}_\text{Entropy: regularization/prior information}
]
$$

(1a)

$z_k$: Latent representation vector of layer k (Encoder output). Encoder_k(x_{k-1}). $x_0$ is preprocessed sensor input.

$\hat{z}_k$: Predicted next-time latent representation in layer k (Predictor output). Predictor_k(z_{k-1}, c_k'). Context c_k' may include lower layer information or time information.

$Target(z_k)$: Actual latent representation calculated at the next time point $t+\Delta t_k$ (using a non-learning target encoder). The prediction target.

$D_{pred}(z_k, \hat{z}_k)$: Prediction loss function. E.g., $|\text{sg}(Target(z_k)) - \hat{z}_k|_2^2$. sg is stop-gradient.

$q(z_k|c_k)$: Inference distribution of latent representation z_k given the current context c_k (e.g., z_{k-1}) (encoder models this).

$p(z_k)$: Prior distribution of latent representation z_k (e.g., standard normal distribution N(0, I)).

$\beta_k$: Hyperparameter that adjusts the weight between the energy term and entropy term.

The overall free energy objective is the sum of the free energies of each layer, weighted by weights ($\lambda_k(t)$) adjusted by the meta-controller:

$$
F_{JEPA}(t) = \sum_{k=1}^{K} \lambda_k(t) F_k(t)
$$

(1b)

$\lambda_k(t) \in [10^{-3}, 1]$: Layer-specific weights dynamically adjusted by the meta-controller (π_meta). Initial value is 1/K.

### 2.2 Continuous-Time Model (CT-LNN) and Learning

CT-LNN dynamics:

$$
\dot{h}(t) = f_\theta(h(t), u(t)) \text{ where } u(t) = \text{concat}(\{z_k(t)\}_{k=1..K}, S_{gwt}(t), a_{t-1})
$$

(2a)

$h(t)$: Hidden state vector of CT-LNN.

$f_\theta$: Neural network with parameters $\theta$ (e.g., Liquid Time-Constant Networks, Gated Recurrent Unit ODE).

$u(t)$: Control input at the current time point. Includes hierarchical representations, workspace state, and previous action.

Learning: CT-LNN primarily contributes to representation learning for task performance. It can therefore be learned by backpropagating the loss (L_task) of the task policy network π_task, or it can have its own short-term prediction loss (L_lnn_pred). Backpropagation using the Adjoint Sensitivity Method (similar to original Eq. 2, but with a different objective loss):

$$
\nabla_\theta L_{task} = \int_{t_0}^{t_1} \bar{\lambda}(t') \left( \frac{\partial L_{task}}{\partial h(t')} \frac{\partial h(t')}{\partial \theta} + \frac{\partial L_{task}}{\partial u(t')} \frac{\partial u(t')}{\partial h(t')} \frac{\partial h(t')}{\partial \theta} \right) dt'
$$

(2b)