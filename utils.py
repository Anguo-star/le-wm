import math
import os
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from stable_pretraining import data as dt
from lightning.pytorch.callbacks import Callback


def resolve_h5_dataset_path(name: str, cache_dir=None) -> Path:
    """Find ``<name>.h5`` under STABLEWM_HOME in either layout.

    Two swm versions disagree on the on-disk layout:

    * **0.0.6 wheel**: ``<STABLEWM_HOME>/<name>.h5`` (flat).
    * **Post-PR-#221 source**: ``<STABLEWM_HOME>/datasets/<name>.h5``
      (hard-coded ``sub_folder='datasets'``).

    This helper checks both candidate paths and returns whichever exists,
    so train.py / train_pldm.py can pass ``path=`` to ``HDF5Dataset`` and
    bypass the hard-coded sub_folder logic in the source-overlay version
    while still finding 0.0.6-style flat layouts.
    """
    base = Path(cache_dir) if cache_dir else Path(
        os.environ.get("STABLEWM_HOME", Path.home() / ".stable_worldmodel")
    )
    candidates = [
        base / f"{name}.h5",                  # 0.0.6 wheel / flat
        base / "datasets" / f"{name}.h5",      # post-PR-#221 layout
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"HDF5 dataset '{name}' not found; tried: {[str(p) for p in candidates]}"
    )


class TransformDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.transform(self.dataset[idx])


def _cfg_get(cfg, key: str, default=None):
    if cfg is None:
        return default
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


class AddNormalizedGaussianNoise:
    """Add Gaussian noise to ImageNet-normalized tensors using pixel-space std.

    Per-frame independent sampling (frames = leading dims before C,H,W):
    - Bernoulli(noise_prob) decides whether the frame gets noise
    - if yes, std ~ Uniform(std_min, std_max) is sampled per frame

    Backward compatible: std_min == std_max with noise_prob == 1.0 is the
    same as a fixed std applied to every frame.
    """

    def __init__(self, std_min, std_max, noise_prob: float = 1.0):
        self.std_low = float(std_min)
        self.std_high = float(std_max)
        self.noise_prob = float(noise_prob)
        if self.std_low < 0 or self.std_high < 0:
            raise ValueError("noise std must be non-negative")
        if self.std_low > self.std_high:
            raise ValueError(
                f"noise std range must be ordered std_min <= std_max, "
                f"got ({std_min}, {std_max})"
            )
        if not 0.0 <= self.noise_prob <= 1.0:
            raise ValueError(
                f"noise_prob must be in [0, 1], got {noise_prob}"
            )
        stats = dt.dataset_stats.ImageNet
        channel_std = stats["std"] if isinstance(stats, dict) else stats.std
        self.channel_std = torch.as_tensor(channel_std, dtype=torch.float32)

    @property
    def max_std(self) -> float:
        return self.std_high if self.noise_prob > 0 else 0.0

    def _sample_per_frame_std(self, leading_shape, device, dtype):
        stds = torch.empty(leading_shape, device=device, dtype=dtype).uniform_(
            self.std_low, self.std_high
        )
        if self.noise_prob < 1.0:
            mask = (torch.rand(leading_shape, device=device) < self.noise_prob).to(dtype)
            stds = stds * mask
        return stds

    def __call__(self, x):
        if not torch.is_tensor(x):
            return x
        if self.std_high <= 0 or self.noise_prob <= 0:
            return x

        if x.ndim < 3:
            stds = self._sample_per_frame_std((), x.device, x.dtype)
            return x + torch.randn_like(x) * stds

        leading_shape = x.shape[:-3]  # frame dims before (C, H, W); may be empty
        stds = self._sample_per_frame_std(leading_shape, x.device, x.dtype)
        per_frame_scale = stds.view(*leading_shape, 1, 1, 1)

        if x.shape[-3] == self.channel_std.numel():
            channel_factor = (1.0 / self.channel_std.to(device=x.device, dtype=x.dtype)).view(
                *([1] * len(leading_shape)), -1, 1, 1
            )
            scale = per_frame_scale * channel_factor
        else:
            scale = per_frame_scale

        return x + torch.randn_like(x) * scale


def get_img_preprocessor(source: str, target: str, img_size: int = 224):
    imagenet_stats = dt.dataset_stats.ImageNet
    to_image = dt.transforms.ToImage(**imagenet_stats, source=source, target=target)
    resize = dt.transforms.Resize(img_size, source=source, target=target)
    return dt.transforms.Compose(to_image, resize)


def get_img_noise_transform(cfg, source: str = "pixels", target: str = "pixels"):
    """Build the training-time pixel noise transform from a config block.

    Expected ``cfg.type`` is ``gaussian_noise`` (renamed from ``gaussian``
    on 2026-05-22 to align with the eval-side corruption naming, which
    now includes ``gaussian_blur`` and ``resize`` as sibling families).
    """
    noise_type = _cfg_get(cfg, "type", "gaussian_noise")
    if noise_type == "gaussian":
        noise_type = "gaussian_noise"
    std_min = _cfg_get(cfg, "std_min", 0.0)
    std_max = _cfg_get(cfg, "std_max", 0.0)
    noise_prob = _cfg_get(cfg, "noise_prob", 1.0)
    noise = AddNormalizedGaussianNoise(std_min, std_max, noise_prob=noise_prob)

    if noise.max_std <= 0:
        return None
    if noise_type != "gaussian_noise":
        raise ValueError(f"Unsupported image noise type: {noise_type}")

    return dt.transforms.WrapTorchTransform(
        noise, source=source, target=target
    )


class AddGaussianBlur:
    """Per-frame Gaussian spatial blur, parameterised by *kernel size* in
    pixels. The tensor shape is preserved (the blur is a spectral
    low-pass and does *not* change spatial size).

    User-facing API: ``kernel_size`` is the only knob. The kernel sigma
    is derived from the kernel size by torchvision's auto rule
    (``sigma = 0.3 * ((k - 1) * 0.5 - 1) + 0.8``, the OpenCV formula),
    so users do not have to think about sigma at all.

    Even kernel sizes are rounded up to the next odd value.
    ``kernel_size = 1`` is the identity (no-op).

    Per-frame independent sampling mirrors the noise transform's API:

    - ``Bernoulli(apply_prob)`` decides whether each frame is blurred;
    - if so, ``kernel_size`` is sampled uniformly from the odd integers
      in ``[kernel_size_min, kernel_size_max]``.

    Eval convention is ``kernel_size_min == kernel_size_max`` with
    ``apply_prob = 1.0`` (deterministic blur of every frame).

    Indicative kernel sizes (auto sigma in parentheses):

        kernel_size = 1   ->  no-op
        kernel_size = 3   ->  3x3   (sigma 0.5)   light
        kernel_size = 7   ->  7x7   (sigma 1.1)   mild
        kernel_size = 15  ->  15x15 (sigma 2.3)   moderate
        kernel_size = 31  ->  31x31 (sigma 4.7)   heavy
    """

    def __init__(self, kernel_size_min, kernel_size_max,
                 apply_prob: float = 1.0):
        ks_low = int(kernel_size_min)
        ks_high = int(kernel_size_max)
        if ks_low < 1 or ks_high < 1:
            raise ValueError("kernel_size must be >= 1")
        if ks_low > ks_high:
            raise ValueError(
                "kernel_size range must be ordered: "
                f"got min={kernel_size_min} > max={kernel_size_max}"
            )
        # Force odd; raise the floor / ceiling as needed.
        if ks_low % 2 == 0:
            ks_low += 1
        if ks_high % 2 == 0:
            ks_high += 1
        self.ks_low = ks_low
        self.ks_high = ks_high
        self.apply_prob = float(apply_prob)
        if not 0.0 <= self.apply_prob <= 1.0:
            raise ValueError(f"apply_prob must be in [0, 1], got {apply_prob}")

    @property
    def max_kernel_size(self) -> int:
        return self.ks_high if self.apply_prob > 0 else 1

    @staticmethod
    def _auto_sigma(ks: int) -> float:
        """Standard OpenCV / torchvision auto-sigma rule. Always
        computed explicitly so the call does not depend on the runtime
        torchvision version defaulting ``sigma=None`` for us.
        ``sigma = 0.3 * ((k - 1) * 0.5 - 1) + 0.8``; for k>=3 this is
        well-defined and positive."""
        return max(0.3 * ((ks - 1) * 0.5 - 1) + 0.8, 0.1)

    def __call__(self, x):
        if not torch.is_tensor(x) or x.ndim < 3:
            return x
        if self.ks_high <= 1 or self.apply_prob <= 0:
            return x

        from torchvision.transforms.v2.functional import gaussian_blur as _gblur

        # Fast path: deterministic kernel size (eval convention).
        if self.ks_low == self.ks_high and self.apply_prob >= 1.0:
            ks = self.ks_high
            if ks <= 1:
                return x
            sigma = self._auto_sigma(ks)
            return _gblur(x, kernel_size=[ks, ks], sigma=[sigma, sigma])

        # Slow path: per-frame stochastic kernel size (training-time).
        leading_shape = x.shape[:-3]
        n_frames = 1
        for d in leading_shape:
            n_frames *= int(d)
        x_flat = x.reshape(n_frames, *x.shape[-3:])
        ks_choices = list(range(self.ks_low, self.ks_high + 1, 2))  # odd only
        idx = torch.randint(0, len(ks_choices), (n_frames,), device=x.device)
        if self.apply_prob < 1.0:
            apply_mask = torch.rand(n_frames, device=x.device) < self.apply_prob
        else:
            apply_mask = None
        out = x_flat.clone()
        for i in range(n_frames):
            if apply_mask is not None and not bool(apply_mask[i]):
                continue
            ks = ks_choices[int(idx[i].item())]
            if ks <= 1:
                continue
            sigma = self._auto_sigma(ks)
            out[i:i + 1] = _gblur(
                x_flat[i:i + 1], kernel_size=[ks, ks], sigma=[sigma, sigma]
            )
        return out.reshape(*x.shape)


class AddResize:
    """Two-step bilinear corruption that destroys high-frequency detail
    without changing the tensor shape:

        Step 1 (downscale):  H x W  ->  round(H*factor) x round(W*factor)
        Step 2 (upscale):    round(H*factor) x round(W*factor)  ->  H x W

    The output shape matches the input shape exactly, so downstream code
    (encoders, dataloaders) sees an unchanged interface; only the spectrum
    of the image has been low-passed by the round-trip through a smaller
    intermediate resolution.

    Per-frame independent sampling mirrors the noise transform's API:

    - ``Bernoulli(apply_prob)`` decides whether each frame is degraded;
    - if so, ``factor ~ Uniform(factor_min, factor_max)`` is sampled.

    ``factor = 1.0`` is a no-op; smaller factors discard more detail.
    Indicative examples on a 224x224 input:

        factor=0.75 -> intermediate 168x168  (mild)
        factor=0.50 -> intermediate 112x112  (moderate)
        factor=0.25 -> intermediate  56x56   (heavy)
        factor=0.10 -> intermediate  22x22   (extreme; only blobs survive)

    The eval convention is ``factor_min == factor_max`` with
    ``apply_prob = 1.0`` (deterministic round-trip on every frame).
    """

    def __init__(self, factor_min, factor_max, apply_prob: float = 1.0):
        self.factor_low = float(factor_min)
        self.factor_high = float(factor_max)
        self.apply_prob = float(apply_prob)
        if not 0 < self.factor_low <= self.factor_high <= 1.0:
            raise ValueError(
                "resize factor must satisfy 0 < min <= max <= 1, "
                f"got ({factor_min}, {factor_max})"
            )
        if not 0.0 <= self.apply_prob <= 1.0:
            raise ValueError(f"apply_prob must be in [0, 1], got {apply_prob}")

    @property
    def max_resize_strength(self) -> float:
        """0 = no degradation, 1 = full destruction. Useful for diagnostic
        sensitivity sweeps that need a scalar 'amount of corruption'."""
        return (1.0 - self.factor_low) if self.apply_prob > 0 else 0.0

    @staticmethod
    def _resize_one(x_chw, factor: float):
        h, w = x_chw.shape[-2:]
        if factor >= 1.0:
            return x_chw
        h_low = max(1, int(round(h * factor)))
        w_low = max(1, int(round(w * factor)))
        small = F.interpolate(x_chw.unsqueeze(0), size=(h_low, w_low),
                              mode="bilinear", align_corners=False)
        big = F.interpolate(small, size=(h, w),
                            mode="bilinear", align_corners=False)
        return big.squeeze(0)

    def __call__(self, x):
        if not torch.is_tensor(x) or x.ndim < 3:
            return x
        if self.factor_low >= 1.0 or self.apply_prob <= 0:
            return x

        leading_shape = x.shape[:-3]
        n_frames = 1
        for d in leading_shape:
            n_frames *= int(d)
        x_flat = x.reshape(n_frames, *x.shape[-3:])

        # Fast path: deterministic factor (eval convention).
        if self.factor_low == self.factor_high and self.apply_prob >= 1.0:
            factor = self.factor_low
            out = torch.stack(
                [self._resize_one(x_flat[i], factor) for i in range(n_frames)],
                dim=0,
            )
            return out.reshape(*x.shape)

        # Slow path: per-frame stochastic factor.
        factors = torch.empty(n_frames, device=x.device, dtype=x.dtype).uniform_(
            self.factor_low, self.factor_high
        )
        if self.apply_prob < 1.0:
            mask = torch.rand(n_frames, device=x.device) < self.apply_prob
            factors = torch.where(mask, factors, torch.ones_like(factors))
        out = torch.stack(
            [self._resize_one(x_flat[i], float(factors[i].item())) for i in range(n_frames)],
            dim=0,
        )
        return out.reshape(*x.shape)


def build_eval_corruption(cfg):
    """Build an eval-time image-corruption transform from
    ``cfg.eval.corruption``.

    Dispatches on ``cfg.type``; returns ``None`` if the corruption is
    disabled or has a no-op-magnitude parameter.

    Supported types and their parameters:

    - ``gaussian_noise`` (default): additive ImageNet-space noise via
      :class:`AddNormalizedGaussianNoise`. Uses ``std``.
    - ``gaussian_blur``: spatial Gaussian blur via :class:`AddGaussianBlur`.
      Uses ``kernel_size`` (odd integer; sigma is auto-derived).
    - ``resize``: bilinear downscale-then-upscale via :class:`AddResize`.
      Uses ``factor`` (no-op at 1.0).
    """
    if cfg is None:
        return None
    ctype = _cfg_get(cfg, "type", "gaussian_noise")
    if ctype == "gaussian_noise":
        std = float(_cfg_get(cfg, "std", 0.0))
        if std <= 0:
            return None
        return AddNormalizedGaussianNoise(std, std)
    if ctype == "gaussian_blur":
        ks = int(round(float(_cfg_get(cfg, "kernel_size", 0))))
        if ks <= 1:
            return None
        if ks % 2 == 0:
            ks += 1
        return AddGaussianBlur(ks, ks)
    if ctype == "resize":
        factor = float(_cfg_get(cfg, "factor", 1.0))
        if factor >= 1.0:
            return None
        return AddResize(factor, factor)
    raise ValueError(f"Unsupported corruption type: {ctype}")


def make_eval_corruption(magnitude: float, ctype: str = "gaussian_noise"):
    """Build a corruption transform from a scalar magnitude and a type
    tag, intended for diagnostic-probe injection (where we want to
    parameterise the corruption strength as a single number rather than
    a config block).

    Magnitude semantics by type:

    - ``gaussian_noise`` -> noise std (>0 enables)
    - ``gaussian_blur``  -> kernel_size in pixels (>1 enables; rounded
      up to next odd integer)
    - ``resize``         -> downscale factor (<1.0 enables)

    Returns ``None`` when the magnitude is at the no-op value for the
    chosen type, so callers can short-circuit cleanly.
    """
    if ctype == "gaussian_noise":
        if magnitude <= 0:
            return None
        return AddNormalizedGaussianNoise(magnitude, magnitude)
    if ctype == "gaussian_blur":
        ks = int(round(float(magnitude)))
        if ks <= 1:
            return None
        if ks % 2 == 0:
            ks += 1
        return AddGaussianBlur(ks, ks)
    if ctype == "resize":
        if magnitude >= 1.0:
            return None
        return AddResize(magnitude, magnitude)
    raise ValueError(f"Unsupported corruption type: {ctype}")


def corruption_tag(cfg) -> str:
    """Build a filename-safe tag from a corruption config.

    Returns an empty string for an unconfigured or no-op corruption.
    Naming is chosen so blur / resize tags do not collide with the
    existing Gaussian-noise tag ``std<X>`` used throughout the eval
    summary tooling:

    - ``gaussian_noise`` -> ``std<X>``      (filename unchanged from
      pre-rename history; this keeps existing aggregators working)
    - ``gaussian_blur``  -> ``blur_ks<X>``  (kernel size)
    - ``resize``         -> ``rs_factor<X>``
    """
    if cfg is None:
        return ""
    ctype = _cfg_get(cfg, "type", "gaussian_noise")
    if ctype == "gaussian_noise":
        std = float(_cfg_get(cfg, "std", 0.0))
        if std <= 0:
            return ""
        return f"std{std:g}"
    if ctype == "gaussian_blur":
        ks = int(round(float(_cfg_get(cfg, "kernel_size", 0))))
        if ks <= 1:
            return ""
        if ks % 2 == 0:
            ks += 1
        return f"blur_ks{ks}"
    if ctype == "resize":
        factor = float(_cfg_get(cfg, "factor", 1.0))
        if factor >= 1.0:
            return ""
        return f"rs_factor{factor:g}"
    raise ValueError(f"Unsupported corruption type: {ctype}")


def get_column_normalizer(dataset, source: str, target: str):
    """Get normalizer for a specific column in the dataset."""
    col_data = dataset.get_col_data(source)
    data = torch.from_numpy(np.array(col_data))
    data = data[~torch.isnan(data).any(dim=1)]
    mean = data.mean(0, keepdim=True).clone()
    std = data.std(0, keepdim=True).clone()

    def norm_fn(x):
        return ((x - mean) / std).float()

    normalizer = dt.transforms.WrapTorchTransform(norm_fn, source=source, target=target)
    return normalizer


class ModelObjectCallBack(Callback):
    """Callback to pickle model object after each epoch."""

    def __init__(self, dirpath, filename="model_object", epoch_interval: int = 1):
        super().__init__()
        self.dirpath = Path(dirpath)
        self.filename = filename
        self.epoch_interval = epoch_interval

    def on_train_epoch_end(self, trainer, pl_module):
        super().on_train_epoch_end(trainer, pl_module)

        output_path = (
            self.dirpath
            / f"{self.filename}_epoch_{trainer.current_epoch + 1}_object.ckpt"
        )

        if trainer.is_global_zero:
            if (trainer.current_epoch + 1) % self.epoch_interval == 0:
                self._dump_model(pl_module.model, output_path)

            # save final epoch
            if (trainer.current_epoch + 1) == trainer.max_epochs:
                self._dump_model(pl_module.model, output_path)

    def _dump_model(self, model, path):
        try:
            torch.save(model, path)
        except Exception as e:
            print(f"Error saving model object: {e}")
