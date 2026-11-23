# Exchange Symmetry in Multiplicative Vision Interactions

**Interactive Article (GitHub Pages):** [https://nooobkevin.github.io/exchange-symmetry-vision/](https://nooobkevin.github.io/exchange-symmetry-vision/)  
**Precursor Study:** [Revisiting Wedge Products in Subspace Clifford Networks](https://nooobkevin.github.io/CliffordNet/)

When two feature vectors $(u, v)$ interact multiplicatively, every shared-projection bilinear map decomposes into an exchange-symmetric part $s = (p + q)/\sqrt{2}$ and an exchange-antisymmetric part $a = (p - q)/\sqrt{2}$, where $p = (Au)\odot(Bv)$ and $q = (Av)\odot(Bu)$.

This repository tests a concrete question across synthetic bilinear teachers, frozen ConvNeXt-Atto probes, paired ImageNet-1k fine-tuning, 4-way orientation prediction, and 100-epoch from-scratch ConvNeXt-Atto and ViT-Tiny runs on an 8x NVIDIA H800 node:

> **When should multiplicative exchange symmetry be hard-coded (`Anti` or `Sym`), when does unconstrained Hadamard multiplication (`Star`) or single-token quadratic gating (`SelfAvg`) suffice, and what symmetry does gradient descent choose when the angle $\theta$ is learnable (`Free`)?**

## Main findings

### 1. Coordinate alignment, rank doubling, and measurement (`scripts/exp1_*.py`)
- Channel-grouped `k=4` wedge products collapse under random orthogonal basis rotations ($R^2 = 0.999 \to -0.021$), whereas projected `anti` is strictly $O(d)$-invariant ($0.998 \to 0.998$).
- Under rank bottlenecks ($R = d/2$), learnable `free_star` doubles expressible bilinear rank from $R$ to $2R$ using only $R$ scalar angles ($R^2 = 0.967$ vs. $0.855$ for `star` on general 50/50 teachers).
- Because paired channels in `W_out` can cancel antisymmetric components even when per-channel $\theta_r \neq +\pi/4$, post-readout swapped-pair `FuncAnti` is needed to measure actual operator symmetry and calibrates to $\pm 0.001$ across $E_{\text{anti}} \in [0, 1]$.

### 2. Task symmetry, pair geometry, and stage depth
- At Stage 4 (`7x7`, one block before GAP) with horizontally directed pairs on reflection-invariant ImageNet-1k fine-tuning (Wave A-Matched, 4 paired seeds), `anti` is suppressed 10x (`RMS = 0.037` vs. `0.374` for `sym`, even with equalized step-0 RMS), and `free_star` converges to **99.7% symmetry** (`FuncAnti = 0.003`).
- Switching the task on the exact same backbone and images to 4-way rotation prediction `{0, 90, 180, 270}` (Wave C-Matched, 3 paired seeds) reverses every indicator: `anti` grows to `RMS = 0.624`, zeroing `Anti` drops `anti` by `-7.37%` and `free_star` by `-3.51%` (15x the impact of zeroing `Sym`), and `free_sym` escapes pure symmetry (`FuncAnti: 0.000 -> 0.498`).
- Switching Stage-4 pairs to isotropic center-surround `(u, DWConv5x5(u))` on ImageNet-1k (Wave B1, 3 paired seeds) removes the reflection conflict, lifting `anti` to `RMS = 0.417` (`+0.20%` over baseline, `-0.91%` when zeroed).
- Moving the directed pair earlier to mid-level Stage 3 (`14x14`, three nonlinear blocks before GAP; Wave B2, 3 paired seeds) also prevents collapse (`anti RMS = 0.069`) and shifts `free_star` upward into antisymmetry (`FuncAnti = 0.565`, `Anti->0 = -0.23%` vs. `Sym->0 = -0.15%`).
- The same task flip holds inside Vision Transformers (Wave F, `ViT-Tiny` 4-way rotation, 2 paired seeds): switching from ImageNet-1k classification to 4-way rotation flips `free_star` from `FuncAnti = 0.167` (`Sym->0 = -2.88%`) to `0.729` (`Anti->0 = -6.18%` vs. `Sym->0 = -0.84%`, best accuracy `90.35%`).

### 3. Cross-token multiplication vs. single-token quadratic gating (`SelfAvg`)
- `SelfAvg(u, v) = ((Au)*(Bu) + (Av)*(Bv)) / sqrt(2)` matches `Sym` in parameter count, receptive field, and exchange symmetry without multiplying across positions (`SelfAvg - Sym = A(u-v)*B(u-v)/sqrt(2)`).
- Swapping trained `sym` / `free_star` checkpoints to `self_avg` without retraining drops 50k ImageNet-1k Top-1 by only `-0.09% / -0.08%` (`96.1%` prediction agreement) in 30-epoch Stage-4 fine-tuning, where retrained `self_avg` matches `sym` on clean Top-1 (`71.45%` vs. `71.42%`) and trails by `-1.35%` on 0.12x low-contrast images.
- In 100-epoch from-scratch `ConvNeXt-Atto`, the same zero-retraining swap crashes `sym` by `-3.20%` and `free_star` by `-14.15%` (`KL = 0.696`), confirming genuine reliance on cross-token multiplication.

### 4. 100-epoch from-scratch co-adaptation and architectural complementarity
- Training multi-stage `ConvNeXt-Atto` from scratch on ImageNet-1k (Wave D, 24 runs, 3 paired seeds) unlocks `anti` (`Stage-4 RMS = 0.744`, `BypassDelta = -5.13%`). Downstream 7x7 depthwise conv filters increase horizontal antisymmetry **9.7x** (`6.86%` vs. `0.71%`), while quadratic GELU rectification supplies `26.4%` of post-GAP shift energy.
- Inside the same network, mid-level Stage 3 preserves `FuncAnti = 0.375` in `free_star`, whereas pre-GAP Stage 4 drops to `0.049` (`0.132` in `star`).
- Host architecture determines whether cross-token products beat `self_avg`: in `ConvNeXt-Atto` (no self-attention), every cross-token bilinear mode beats `self_avg` (`71.19%`), led by unconstrained `star` (`71.49 +/- 0.27%`, `+0.58%` over `none` at half the products of `sym`/`anti`). In 100-epoch from-scratch `ViT-Tiny` (Wave E, 18 runs, 3 paired seeds), where softmax self-attention already mixes token pairs, single-token quadratic `self_avg` wins across all 3/3 seeds at `67.93 +/- 0.54%` (`+0.64 +/- 0.25%` over `none` `67.29%`, ahead of `star` `67.63%`, `anti` `67.57%`, and `sym` `67.48%`).

## Repository structure

- `src/exsym/`: `ExchangeBilinearCore`, `GroupedWedgeCore`, `SpatialExchangeAdapter`, `ConvNeXtWithAdapter`, `ViTWithAdapter`, `FrozenPairHead`, and zero-decode `/dev/shm` `ShmImageNetDataset`.
- `tests/test_exchange_symmetry.py`: 7 mathematical unit tests verifying swap sign, step-0 forward/gradient equivalence, nonzero $\partial\mathcal{L}/\partial\theta$, exact `s + a` decomposition, and `SelfAvg - Sym` algebraic identity.
- `scripts/`: Synthetic oracle (`exp1_*.py`), frozen probes (`exp2_frozen_probes.py`), ImageNet-1k trainer (`train_in1k.py`), multi-GPU priority scheduler (`run_queue.py`), causal intervention & demodulation forensics (`eval_*.py`), and blog data builder (`export_exsym_blog_builder.py`).
- `results/`: Verified JSON summaries and full 50k corruption/intervention forensics.
- `docs/`: Zero-dependency interactive HTML/SVG article hosted via GitHub Pages.

