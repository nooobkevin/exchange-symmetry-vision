# Exchange Symmetry in Multiplicative Vision Interactions

**Interactive Article (GitHub Pages):** [https://nooobkevin.github.io/exchange-symmetry-vision/](https://nooobkevin.github.io/exchange-symmetry-vision/)  
**Precursor Study:** [Revisiting Wedge Products in Subspace Clifford Networks](https://nooobkevin.github.io/CliffordNet/)

When two feature vectors $(u, v)$ interact multiplicatively, every shared-projection bilinear map decomposes into an exchange-symmetric part $s = (p + q)/\sqrt{2}$ and an exchange-antisymmetric part $a = (p - q)/\sqrt{2}$, where $p = (Au)\odot(Bv)$ and $q = (Av)\odot(Bu)$.

This repository tests a concrete question across synthetic bilinear teachers, frozen ConvNeXt-Atto probes, paired ImageNet-1k fine-tuning, 4-way orientation prediction, and 100-epoch from-scratch ConvNeXt-Atto and ViT-Tiny runs on an 8x NVIDIA H800 node:

> **When should multiplicative exchange symmetry be hard-coded (`Anti` or `Sym`), when does unconstrained Hadamard multiplication (`Star`) or single-token quadratic gating (`SelfAvg`) suffice, and what symmetry does gradient descent choose when the angle $\theta$ is learnable (`Free`)?**

## Headline Findings

1. **Coordinate-alignment & rank-doubling laws (`scripts/exp1_*.py`)**:
   - Hard-coded channel-grouped `k=4` wedge products collapse under random orthogonal basis rotations ($R^2 = 0.999 \to -0.021$), whereas projected `anti` is strictly $O(d)$-invariant ($0.998 \to 0.998$).
   - Under rank bottlenecks ($R = d/2$), learnable `free_star` doubles expressible bilinear rank from $R$ to $2R$ using only $R$ scalar angles ($R^2 = 0.967$ vs. $0.855$ for `star` on general 50/50 teachers).
   - Because paired channels in `W_out` can cancel antisymmetric components even when per-channel $\theta_r \neq +\pi/4$, post-readout swapped-pair `FuncAnti` is required to measure true operator symmetry and calibrates to $\pm 0.001$ across $E_{\text{anti}} \in [0, 1]$.

2. **Task symmetry $\times$ pair geometry double dissociation (ImageNet-1k vs. 4-Way Rotation vs. Center-Context)**:
   - **Directed neighbor + reflection-invariant ImageNet-1k fine-tuning (Wave A-Matched, 4 paired seeds)**: `anti` is suppressed 10x (`RMS = 0.037` vs. `0.374` for `sym`, even with equalized step-0 RMS), and `free_star` converges to **99.7% symmetry** (`FuncAnti = 0.003`).
   - **Directed neighbor + 4-way rotation prediction `{0, 90, 180, 270}` (Wave C-Matched, 3 paired seeds)**: On the exact same backbone and images, `anti` grows to `RMS = 0.624`, zeroing `Anti` drops `anti` by `-7.37%` and `free_star` by `-3.51%` (15x larger than zeroing `Sym`), and `free_sym` actively escapes pure symmetry (`FuncAnti: 0.000 -> 0.498`).
   - **Isotropic center-surround `(u, DWConv5x5(u))` on ImageNet-1k (Wave B1, 3 paired seeds)**: Swapping center and surround is not a spatial reflection, so `anti` is no longer suppressed (`RMS = 0.417`, `+0.20%` over baseline, `-0.91%` when zeroed).

3. **`SelfAvg` audit: cross-token multiplication vs. single-token quadratic gating**:
   - `SelfAvg(u, v) = ((Au)*(Bu) + (Av)*(Bv)) / sqrt(2)` matches `Sym` in parameter count, receptive field, and exchange symmetry without multiplying across positions (`SelfAvg - Sym = A(u-v)*B(u-v)/sqrt(2)`).
   - In 30-epoch ImageNet-1k fine-tuning, `SelfAvg` matches `Sym` and `Free-Star` on clean Top-1 (`71.45%` vs. `71.42%` and `71.47%`), showing short fine-tuning gains come from degree-2 channel gating, though `SelfAvg` loses `-1.35%` on 0.12x low-contrast images.

4. **100-epoch from-scratch co-adaptation & depth-symmetry hierarchy (Wave D ConvNeXt-Atto & Wave E ViT-Tiny)**:
   - **Co-adaptation unlocks `anti`**: From scratch on ImageNet-1k, `anti` grows to `RMS = 0.744` (`BypassDelta = -5.13%`, beating `none` on 3/3 seeds). Downstream 7x7 depthwise conv filters increase horizontal antisymmetry **9.7x** (`6.86%` vs. `0.71%`, demodulating `odd * odd = even`), while quadratic GELU rectification supplies `26.4%` of post-GAP shift energy.
   - **Depth-symmetry hierarchy**: Mid-level Stage 3 (`14x14`) preserves `FuncAnti = 0.375` in `free_star`, whereas pre-GAP Stage 4 (`7x7`) collapses to `0.049`.
   - **Cross-token wins when the backbone lacks pairwise token mixing**: In from-scratch `ConvNeXt-Atto`, every cross-token bilinear mode beats `self_avg`, led by unconstrained `star` (`71.49 +/- 0.27%`, `+0.58%` over `none`, `+0.30%` over `self_avg`, `+3.61%` on low-contrast at half the products of `sym`/`anti`).

## Repository Structure

- `src/exsym/`: `ExchangeBilinearCore`, `GroupedWedgeCore`, `SpatialExchangeAdapter`, `ConvNeXtWithAdapter`, `ViTWithAdapter`, `FrozenPairHead`, and zero-decode `/dev/shm` `ShmImageNetDataset`.
- `tests/test_exchange_symmetry.py`: 7 mathematical unit tests verifying swap sign, step-0 forward/gradient equivalence, nonzero $\partial\mathcal{L}/\partial\theta$, exact `s + a` decomposition, and `SelfAvg - Sym` algebraic identity.
- `scripts/`: Synthetic oracle (`exp1_*.py`), frozen probes (`exp2_frozen_probes.py`), ImageNet-1k trainer (`train_in1k.py`), multi-GPU priority scheduler (`run_queue.py`), causal intervention & demodulation forensics (`eval_*.py`), and blog data builder (`export_exsym_blog_builder.py`).
- `results/`: Verified JSON summaries and full 50k corruption/intervention forensics.
- `docs/`: Zero-dependency interactive HTML/SVG article hosted via GitHub Pages.

