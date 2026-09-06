"""Ops for the action VAE: dataset-level latent whitening and raw-action normalization.

Mirrors ``ltx_core.model.video_vae.ops`` / ``ltx_core.model.audio_vae.ops`` so that the
action stream obeys the same encode -> normalize contract as the video and audio streams.
"""

from __future__ import annotations

import torch
from torch import nn


class PerChannelStatistics(nn.Module):
    """Per-channel statistics for normalizing / denormalizing the action latent.

    Same buffer names and same semantics as the video and audio VAEs: the statistics are
    computed over the posterior *means* across the entire training split and stored in the
    model's checkpoint under the VAE state_dict. Defaults are identity (std=1, mean=0).

    Difference from the video/audio versions: action latents are channel-last sequences
    ``(B, T, C)`` rather than ``(B, C, F, H, W)``, so the broadcast view is ``(1, 1, -1)``.
    """

    def __init__(self, latent_channels: int = 7):
        super().__init__()
        self.register_buffer("std-of-means", torch.ones(latent_channels))
        self.register_buffer("mean-of-means", torch.zeros(latent_channels))

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        mean = self.get_buffer("mean-of-means").view(1, 1, -1).to(x)
        std = self.get_buffer("std-of-means").view(1, 1, -1).to(x)
        return (x - mean) / std

    def un_normalize(self, x: torch.Tensor) -> torch.Tensor:
        mean = self.get_buffer("mean-of-means").view(1, 1, -1).to(x)
        std = self.get_buffer("std-of-means").view(1, 1, -1).to(x)
        return (x * std) + mean

    @torch.no_grad()
    def fit(self, means: torch.Tensor, std_floor: float = 1e-4) -> None:
        """Fit the buffers from posterior means over the training split.

        Args:
            means: (N, C) posterior means, un-normalized.
            std_floor: lower bound on the per-channel std. A latent dimension that has
                collapsed to the prior has near-constant means, so its std is ~0 and
                ``normalize`` would divide by it and amplify numerical noise. The floor
                makes that failure mode loud (the dimension stays near zero) rather than
                explosive. If the floor is ever hit, the latent is too wide.
        """
        if means.ndim != 2:
            raise ValueError(f"expected (N, C) posterior means, got shape {tuple(means.shape)}")
        self.get_buffer("mean-of-means").copy_(means.mean(dim=0))
        self.get_buffer("std-of-means").copy_(means.std(dim=0).clamp_min(std_floor))


class ActionProcessor(nn.Module):
    """Maps raw actions to [-1, 1] per dimension, and back.

    This is the action analogue of ``AudioProcessor``: it ships inside the VAE so that the
    finetune applies exactly the scaling the VAE was trained with, and so the decoder hands
    back actions in real units (mm, radians, gripper counts) ready to command the arm.

    Two kinds of dimension, and the difference is whether the quantity has a physical bound:

    * The six pose deltas have no physical maximum -- the largest is simply the fastest
      motion that happened to be recorded. Their scale is fitted from the training split's
      min/max and nothing is clamped, so a fast motion in held-out data lands slightly
      outside [-1, 1] rather than being flattened onto the boundary.
    * The gripper has a real travel range (0-850 on the xArm). That constant is used
      directly, exactly as the video VAE divides pixels by 255 rather than by a dataset
      statistic, and it *is* clamped, because a command outside the travel is meaningless.

    Fitting min/max rather than quantiles is only safe because the recorded deltas have no
    sensor spikes: max/p99 runs 2.6-3.8x with p99.9 sitting smoothly in between. On data
    with glitches a single bad frame would set the scale and squash everything else, and
    the fit would need quantiles instead. ``fit`` reports both so this stays checkable.
    """

    def __init__(self, action_dim: int = 7):
        super().__init__()
        self.action_dim = action_dim
        self.register_buffer("range-low", torch.full((action_dim,), -1.0))
        self.register_buffer("range-high", torch.full((action_dim,), 1.0))
        # 1.0 marks a dimension whose range is a physical bound, so values outside it are
        # not possible and are clamped rather than represented.
        self.register_buffer("clamp-mask", torch.zeros(action_dim))

    def _lo_span_mask(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        lo = self.get_buffer("range-low").view(1, 1, -1).to(x)
        hi = self.get_buffer("range-high").view(1, 1, -1).to(x)
        mask = self.get_buffer("clamp-mask").view(1, 1, -1).to(x) > 0.5
        return lo, (hi - lo).clamp_min(1e-8), mask

    def normalize(self, actions: torch.Tensor) -> torch.Tensor:
        """Raw actions (B, T, action_dim) -> [-1, 1]. Clamped only on bounded dimensions."""
        lo, span, mask = self._lo_span_mask(actions)
        normalized = 2.0 * (actions - lo) / span - 1.0
        return torch.where(mask, normalized.clamp(-1.0, 1.0), normalized)

    def un_normalize(self, normalized: torch.Tensor) -> torch.Tensor:
        """[-1, 1] -> raw action units. Bounded dimensions cannot leave their range."""
        lo, span, mask = self._lo_span_mask(normalized)
        normalized = torch.where(mask, normalized.clamp(-1.0, 1.0), normalized)
        return (normalized + 1.0) / 2.0 * span + lo

    @torch.no_grad()
    def fit(
        self,
        actions: torch.Tensor,
        gripper_range: tuple[float, float] | None = (0.0, 850.0),
        gripper_index: int = -1,
    ) -> dict[str, list[float]]:
        """Fit the per-dimension range on the training split.

        Args:
            actions: (N, action_dim) raw actions.
            gripper_range: physical (low, high) travel for the gripper dimension, used
                instead of the fitted min/max and clamped. ``None`` fits it like the rest.
            gripper_index: which dimension the gripper occupies.

        Returns:
            Per-dimension ``min``, ``max`` and ``p99`` of the raw data, for reporting. The
            ratio ``max / p99`` is the check on whether min/max is a safe choice.
        """
        if actions.ndim != 2:
            raise ValueError(f"expected (N, action_dim) actions, got shape {tuple(actions.shape)}")
        actions = actions.to(torch.float64)

        lo = actions.min(dim=0).values
        hi = actions.max(dim=0).values
        p99 = torch.quantile(actions.abs(), 0.99, dim=0)
        mask = torch.zeros(self.action_dim, dtype=torch.float32)

        if gripper_range is not None:
            index = gripper_index % self.action_dim
            lo[index] = gripper_range[0]
            hi[index] = gripper_range[1]
            mask[index] = 1.0

        self.get_buffer("range-low").copy_(lo.to(torch.float32))
        self.get_buffer("range-high").copy_(hi.to(torch.float32))
        self.get_buffer("clamp-mask").copy_(mask)

        return {
            "min": actions.min(dim=0).values.tolist(),
            "max": actions.max(dim=0).values.tolist(),
            "p99": p99.tolist(),
        }
