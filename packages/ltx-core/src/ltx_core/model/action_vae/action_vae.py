"""Per-step action VAE for the World Action Model.

One action (translation delta, rotation delta, gripper) becomes one latent token, matching
the way the video VAE turns one latent cell into one token. The VAE is trained separately
and frozen for the WAM finetune; only ``action_in`` / ``action_out`` (the projections to and
from the DiT width) are trained during the finetune, so the latent width is chosen for the
health of the VAE itself, not to match anything in the transformer.

The defaults -- latent 7, hidden 64, 2 layers -- are the settled configuration; the sweep that
produced them, and the reasoning for beta, are recorded in ``scripts/train_action_vae.py``.
Latent 7 is the data's intrinsic dimensionality: exactly 7 channels stay active whatever width
the model is given, and PCA on the normalized actions confirms full rank 7.

``action_in`` on the transformer side is a single ``Linear``, matching how LTX's video and audio
streams enter through ``patchify_proj`` and ``audio_patchify_proj``. There is no modality marker
-- see ``LTXModel._init_action`` for why one would be redundant.

Contract, identical to ``VideoEncoder`` and ``AudioEncoder``:
  * ``ActionEncoder.forward`` returns *normalized posterior means only*. The log-variance is
    a training-time quantity and is discarded at inference.
  * The dataset-level whitening lives in a ``PerChannelStatistics`` submodule, so it travels
    inside the state_dict under ``per_channel_statistics.*``.
"""

from __future__ import annotations

import torch
from torch import nn

from ltx_core.model.action_vae.ops import ActionProcessor, PerChannelStatistics
from ltx_core.model.disposable import Disposable


def _mlp(in_dim: int, hidden: int, out_dim: int, num_layers: int) -> nn.Sequential:
    """MLP with ``num_layers`` hidden layers of width ``hidden`` and SiLU activations."""
    if num_layers < 1:
        raise ValueError(f"num_layers must be >= 1, got {num_layers}")
    layers: list[nn.Module] = [nn.Linear(in_dim, hidden), nn.SiLU()]
    for _ in range(num_layers - 1):
        layers += [nn.Linear(hidden, hidden), nn.SiLU()]
    layers.append(nn.Linear(hidden, out_dim))
    return nn.Sequential(*layers)


class ActionEncoder(nn.Module, Disposable):
    """Encode raw per-step actions into normalized latent tokens.

    Args:
        action_dim: raw action width (3 translation + 3 rotation vector + 1 gripper = 7).
        latent_channels: latent width per action token.
        hidden: hidden width of the encoder MLP.
        num_layers: number of hidden layers.
    """

    def __init__(
        self,
        action_dim: int = 7,
        latent_channels: int = 7,
        hidden: int = 64,
        num_layers: int = 2,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.latent_channels = latent_channels
        self.action_processor = ActionProcessor(action_dim=action_dim)
        # double_z: the head emits means and log-variance, per channel, as the audio VAE does.
        self.net = _mlp(action_dim, hidden, 2 * latent_channels, num_layers)
        self.per_channel_statistics = PerChannelStatistics(latent_channels=latent_channels)

    def encode_moments(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Raw actions (B, T, action_dim) -> un-normalized posterior (means, logvar).

        Training-time entry point. ``logvar`` is clamped to the range used by
        ``DiagonalGaussianDistribution`` so the KL term cannot blow up early in training.
        """
        normalized = self.action_processor.normalize(actions)
        moments = self.net(normalized)
        means, logvar = torch.chunk(moments, 2, dim=-1)
        return means, logvar.clamp(-30.0, 20.0)

    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        """Raw actions (B, T, action_dim) -> normalized latent means (B, T, latent_channels).

        Inference entry point, and what the WAM finetune calls. Deterministic: the posterior
        log-variance is discarded, exactly as the video and audio encoders do.
        """
        means, _ = self.encode_moments(actions)
        return self.per_channel_statistics.normalize(means)


class ActionDecoder(nn.Module, Disposable):
    """Decode normalized latent tokens back into raw per-step actions."""

    def __init__(
        self,
        action_dim: int = 7,
        latent_channels: int = 7,
        hidden: int = 64,
        num_layers: int = 2,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.latent_channels = latent_channels
        self.per_channel_statistics = PerChannelStatistics(latent_channels=latent_channels)
        self.net = _mlp(latent_channels, hidden, action_dim, num_layers)
        self.action_processor = ActionProcessor(action_dim=action_dim)

    def decode_normalized(self, latent: torch.Tensor) -> torch.Tensor:
        """Un-normalized latent (B, T, latent_channels) -> actions in the [-1, 1] domain.

        Training-time entry point: during VAE training the latent is a reparameterized
        sample, which lives in the un-normalized space (the whitening buffers are only
        fitted once training is finished).

        No tanh on the output. The delta dimensions are scaled by the training split's
        min/max, so held-out data can legitimately fall outside [-1, 1] and a tanh would
        flatten exactly the fast motions we chose not to clip. The gripper is bounded
        instead by ``ActionProcessor.un_normalize``, which clamps only that dimension.
        """
        return self.net(latent)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """Normalized latent (B, T, latent_channels) -> raw actions (B, T, action_dim)."""
        latent = self.per_channel_statistics.un_normalize(latent)
        return self.action_processor.un_normalize(self.decode_normalized(latent))


def reparameterize(means: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Sample from the diagonal Gaussian posterior."""
    return means + torch.randn_like(means) * torch.exp(0.5 * logvar)


def kl_divergence(means: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Per-element KL to a standard normal, shape (B, T, latent_channels), in nats."""
    return 0.5 * (means.pow(2) + logvar.exp() - 1.0 - logvar)
