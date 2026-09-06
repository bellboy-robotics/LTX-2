"""Load a trained action VAE from the checkpoint ``train_action_vae.py`` writes.

The checkpoint is a single ``action_vae.safetensors`` holding both halves under prefixes::

    action_vae.encoder.*                  the encoder MLP
    action_vae.decoder.*                  the decoder MLP
    action_vae.per_channel_statistics.*   latent whitening, shared by both
    action_vae.action_processor.*         raw-action scaling, shared by both

alongside a ``config.json`` carrying the shapes. The prefix layout mirrors how LTX packs its
video and audio VAEs into one file, so the same ``SDOps``-style filtering applies if the action
VAE is ever merged into the main checkpoint.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from ltx_core.model.action_vae.action_vae import ActionDecoder, ActionEncoder

ENCODER_PREFIX = "action_vae.encoder."
DECODER_PREFIX = "action_vae.decoder."
STATISTICS_PREFIX = "action_vae.per_channel_statistics."
PROCESSOR_PREFIX = "action_vae.action_processor."


def _read(checkpoint: Path) -> tuple[dict, dict[str, torch.Tensor]]:
    """Return the config and the raw state dict for an action VAE directory or file."""
    checkpoint = Path(checkpoint)
    directory = checkpoint.parent if checkpoint.is_file() else checkpoint
    weights = checkpoint if checkpoint.is_file() else directory / "action_vae.safetensors"
    if not weights.is_file():
        raise FileNotFoundError(f"no action_vae.safetensors at {weights}")
    config_path = directory / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"no config.json beside {weights}; it carries the model shapes")
    return json.loads(config_path.read_text()), load_file(str(weights))


def _slice(state: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    return {key[len(prefix) :]: value for key, value in state.items() if key.startswith(prefix)}


def load_action_encoder(checkpoint: str | Path, device: str | torch.device = "cpu") -> ActionEncoder:
    """Build the encoder and load its weights. Returned frozen and in eval mode.

    Frozen because the WAM finetune treats the action VAE as fixed: it is used once, at
    precompute, to turn raw actions into whitened latents.
    """
    config, state = _read(Path(checkpoint))
    encoder = ActionEncoder(
        action_dim=config["action_dim"],
        latent_channels=config["latent_channels"],
        hidden=config["hidden"],
        num_layers=config["num_layers"],
    )
    encoder.net.load_state_dict(_slice(state, ENCODER_PREFIX))
    encoder.per_channel_statistics.load_state_dict(_slice(state, STATISTICS_PREFIX))
    encoder.action_processor.load_state_dict(_slice(state, PROCESSOR_PREFIX))
    encoder.eval().requires_grad_(False)
    return encoder.to(device)


def load_action_decoder(checkpoint: str | Path, device: str | torch.device = "cpu") -> ActionDecoder:
    """Build the decoder and load its weights. Returned frozen and in eval mode.

    Needed at deploy, to turn the model's latent prediction back into millimetres and radians,
    and by validation if you want the action error reported in physical units rather than in
    whitened latent space.
    """
    config, state = _read(Path(checkpoint))
    decoder = ActionDecoder(
        action_dim=config["action_dim"],
        latent_channels=config["latent_channels"],
        hidden=config["hidden"],
        num_layers=config["num_layers"],
    )
    decoder.net.load_state_dict(_slice(state, DECODER_PREFIX))
    decoder.per_channel_statistics.load_state_dict(_slice(state, STATISTICS_PREFIX))
    decoder.action_processor.load_state_dict(_slice(state, PROCESSOR_PREFIX))
    decoder.eval().requires_grad_(False)
    return decoder.to(device)
