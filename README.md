# Exchange Symmetry in Multiplicative Vision Interactions

When two feature vectors $(u, v)$ interact multiplicatively, every bilinear map decomposes into an exchange-symmetric part $(p + q)/\sqrt{2}$ and an exchange-antisymmetric part $(p - q)/\sqrt{2}$, where $p = (Au)\odot(Bv)$ and $q = (Av)\odot(Bu)$.

This repository investigates a narrow, falsifiable question motivated by [Subspace Clifford Networks](https://nooobkevin.github.io/CliffordNet/) and the broader Hadamard/bilinear vision literature (StarNet, AFBO, PolyNeXt):

> **What is actually gained or lost by hard-coding exchange antisymmetry (as in Clifford wedge products) versus exchange symmetry versus unconstrained Hadamard multiplication (`Star`), and when the symmetry angle $\theta$ is made learnable (`Free`), what symmetry does gradient descent actually choose?**

## Unified Parameterization

For each interaction channel $r \in \{1, \dots, R\}$:

$$f_r(u, v) = \cos\theta_r \, p_r + \sin\theta_r \, q_r = \alpha_r^{\text{sym}} s_r + \alpha_r^{\text{anti}} a_r$$

- **`anti`** ($\theta = -\pi/4$ fixed): exact exchange-antisymmetric, $f(v, u) = -f(u, v)$.
- **`sym`** ($\theta = +\pi/4$ fixed): exact exchange-symmetric, $f(v, u) = +f(u, v)$.
- **`star`** ($\theta = 0$ fixed): ordinary Hadamard / Star product $p = (Au)\odot(Bv)$.
- **`free_anti` / `free_star` / `free_sym`**: learnable $\theta \in \mathbb{R}^R$ initialized at $-\pi/4$, $0$, or $+\pi/4$. At step 0, `free_anti` and `anti` are identical in forward outputs and input/weight gradients, while $\partial f / \partial \theta$ is strictly nonzero.

Because `SpatialExchangeAdapter` has no internal pointwise activation (pure bilinear residual placed before downstream nonlinear backbone blocks), the residual update $\delta(u, v)$ preserves exact swap symmetry all the way through `W_out`, allowing exact post-`W_out` functional antisymmetry measurement $\mathrm{FuncAnti}(\delta)$ and exact inference-time decomposition sweeps.

## Experiments

1. **Experiment 1 (Synthetic Bilinear Oracle & Phase Diagram)**: Ground-truth teachers across symmetries (`antisymmetric`, `symmetric`, `general`), coordinate alignment (`aligned` vs. random orthogonal rotation $Q$), sample size $N$, rank bottleneck $R$, and noise $\sigma$, plus the Symmetry Thermometer calibration (`exp1b`).
2. **Experiment 2 (Frozen ConvNeXt-Atto Multi-Shot & Directional Probes)**: Stage-3 & Stage-4 frozen features on ImageNet-1k (10/50 shots/class + corruptions) vs. direction-sensitive 4-way rotation classification (`0/90/180/270`).
3. **Experiment 3 (End-to-End ImageNet-1k Fine-Tuning, From-Scratch, Directional Task & Cross-Backbone ViT-Tiny)**: Paired-seed runs on 8x H800 with per-epoch `EffAnti(theta)` and `FuncAnti(delta)` trajectories, corruptions, horizontal-flip consistency, and decomposed inference interventions.

*Experiments are actively running; full results and interactive report will be published to GitHub Pages upon completion.*
