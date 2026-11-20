"""Zero-JPEG-decode /dev/shm ImageNet-1k data loader + corruption & directional transforms.

Guarantees paired runs (different models sharing the same data_seed) observe identical
sample orderings and identical GPU-side augmentations when desired.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms.v2 as T2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ShmImageNetDataset(Dataset):
    """Read-only np.memmap dataset over /dev/shm/in1k_raw_cache_ar320.

    All parallel processes share the exact same physical tmpfs pages with zero copy and
    zero JPEG decoding overhead.
    """

    def __init__(
        self,
        cache_dir: str | Path = "/dev/shm/in1k_raw_cache_ar320",
        split: str = "train",
        image_size: int = 224,
        hflip: bool = True,
        rrc_scale: tuple[float, float] = (0.08, 1.0),
        color_jitter: float = 0.0,
        rotation_task: bool = False,
    ) -> None:
        super().__init__()
        self.cache_dir = Path(cache_dir)
        self.split = split
        self.image_size = image_size
        self.hflip = hflip
        self.rrc_scale = rrc_scale
        self.rotation_task = rotation_task

        prefix = "train" if split == "train" else "val"
        self.img_path = str(self.cache_dir / f"{prefix}_images.npy")
        self.lbl_path = str(self.cache_dir / f"{prefix}_labels.npy")
        self.shp_path = str(self.cache_dir / f"{prefix}_shapes.npy") if split == "train" else None

        # Open lazily per worker process to avoid pickle bloat
        self._imgs = None
        self._lbls = None
        self._shps = None

        lbls = np.load(self.lbl_path, mmap_mode="r")
        self._len = int(len(lbls))
        del lbls

        self.color_jitter = (
            T2.ColorJitter(brightness=color_jitter, contrast=color_jitter, saturation=color_jitter)
            if (split == "train" and color_jitter > 0.0)
            else None
        )

    def _ensure_open(self) -> None:
        if self._imgs is None:
            self._imgs = np.load(self.img_path, mmap_mode="r")
            self._lbls = np.load(self.lbl_path, mmap_mode="r")
            if self.shp_path is not None and Path(self.shp_path).exists():
                self._shps = np.load(self.shp_path, mmap_mode="r")

    def __len__(self) -> int:
        return self._len

    def _random_resized_crop(self, img: torch.Tensor) -> torch.Tensor:
        _, h, w = img.shape
        area = float(h * w)
        log_ratio = (math.log(3.0 / 4.0), math.log(4.0 / 3.0))
        s_min, s_max = self.rrc_scale
        top, left, ch, cw = 0, 0, h, w
        found = False
        for _ in range(10):
            target_area = area * float(torch.empty(1).uniform_(s_min, s_max).item())
            aspect = math.exp(float(torch.empty(1).uniform_(log_ratio[0], log_ratio[1]).item()))
            cw_t = int(round(math.sqrt(target_area * aspect)))
            ch_t = int(round(math.sqrt(target_area / aspect)))
            if 0 < cw_t <= w and 0 < ch_t <= h:
                top = int(torch.randint(0, h - ch_t + 1, (1,)).item())
                left = int(torch.randint(0, w - cw_t + 1, (1,)).item())
                ch, cw = ch_t, cw_t
                found = True
                break
        if not found:
            in_ratio = float(w) / float(h)
            if in_ratio < 3.0 / 4.0:
                cw = w
                ch = int(round(cw / (3.0 / 4.0)))
            elif in_ratio > 4.0 / 3.0:
                ch = h
                cw = int(round(ch * (4.0 / 3.0)))
            else:
                ch, cw = h, w
            top = (h - ch) // 2
            left = (w - cw) // 2
        crop = img[:, top : top + ch, left : left + cw]
        return F.interpolate(
            crop.unsqueeze(0).float(),
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        ).squeeze(0).clamp_(0.0, 255.0).to(torch.uint8)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        self._ensure_open()
        if self.split == "train":
            if self._shps is not None:
                nh, nw = int(self._shps[idx, 0]), int(self._shps[idx, 1])
                arr = np.array(self._imgs[idx, :, :nh, :nw], copy=True)
            else:
                arr = np.array(self._imgs[idx], copy=True)
            img = torch.from_numpy(arr)
            img = self._random_resized_crop(img)
            if self.hflip and not self.rotation_task and torch.rand(1).item() < 0.5:
                img = torch.flip(img, dims=[2])
            if self.color_jitter is not None:
                img = self.color_jitter(img)
        else:
            arr = np.array(self._imgs[idx], copy=True)
            img = torch.from_numpy(arr)

        if self.rotation_task:
            # Deterministic per-index rotation on val, random 4-way {0, 90, 180, 270} on train
            if self.split == "train":
                k = int(torch.randint(0, 4, (1,)).item())
            else:
                k = idx % 4
            if k > 0:
                img = torch.rot90(img, k=k, dims=[1, 2])
            return img, k

        label = int(self._lbls[idx])
        return img, label


def normalize_batch_gpu(
    imgs_u8: torch.Tensor,
    mean_t: torch.Tensor,
    std_t: torch.Tensor,
) -> torch.Tensor:
    """Convert uint8 (B, 3, H, W) tensor on GPU to normalized float32."""
    return (imgs_u8.float().div_(255.0) - mean_t) / std_t


def apply_corruption_gpu(
    x_norm: torch.Tensor,
    corruption: str,
    mean_t: torch.Tensor,
    std_t: torch.Tensor,
    seed: int = 0,
) -> torch.Tensor:
    """Apply standardized OOD/diagnostic perturbations in [0, 1] pixel space then re-normalize."""
    if corruption == "clean":
        return x_norm
    if corruption == "hflip":
        return torch.flip(x_norm, dims=[3])

    x_pix = (x_norm * std_t + mean_t).clamp(0.0, 1.0)
    if corruption == "contrast_012":
        mean_pix = x_pix.mean(dim=(2, 3), keepdim=True)
        x_pix = ((x_pix - mean_pix) * 0.12 + mean_pix).clamp(0.0, 1.0)
    elif corruption == "gaussian_noise":
        gen = torch.Generator(device=x_pix.device).manual_seed(seed)
        noise = torch.randn(x_pix.shape, generator=gen, device=x_pix.device, dtype=x_pix.dtype) * 0.18
        x_pix = (x_pix + noise).clamp(0.0, 1.0)
    elif corruption == "fft_phase":
        # Replace amplitude spectrum with flat unit amplitude while preserving exact phase
        freq = torch.fft.rfft2(x_pix.float(), dim=(-2, -1))
        phase = torch.angle(freq)
        recon = torch.fft.irfft2(torch.polar(torch.ones_like(phase), phase), s=x_pix.shape[-2:], dim=(-2, -1))
        # Normalize per-image/channel to [0, 1] matching CliffordNet eval_corruptions.py
        mn = recon.amin(dim=(-2, -1), keepdim=True)
        mx = recon.amax(dim=(-2, -1), keepdim=True)
        x_pix = ((recon - mn) / (mx - mn + 1e-6)).to(x_norm.dtype)
    else:
        raise ValueError(f"Unknown corruption: {corruption!r}")
    return (x_pix - mean_t) / std_t


def make_stratified_indices(
    labels: np.ndarray,
    shots_per_class: int,
    seed: int = 42,
) -> np.ndarray:
    """Deterministic class-stratified sample index list for ImageNet few-shot probes."""
    rng = np.random.default_rng(seed)
    num_classes = int(labels.max()) + 1
    chosen: list[np.ndarray] = []
    for c in range(num_classes):
        idx_c = np.flatnonzero(labels == c)
        perm = rng.permutation(idx_c)
        chosen.append(perm[: min(shots_per_class, len(perm))])
    all_idx = np.concatenate(chosen)
    rng.shuffle(all_idx)
    return all_idx
