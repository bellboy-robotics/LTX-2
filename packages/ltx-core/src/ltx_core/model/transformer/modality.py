from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Modality:
    """
    Input data for a single modality (video or audio) in the transformer.
    Bundles the latent tokens, timestep embeddings, positional information,
    and text conditioning context for processing by the diffusion transformer.
    Attributes:
        latent: Patchified latent tokens, shape ``(B, T, D)`` where *B* is
            the batch size, *T* is the total number of tokens (noisy +
            conditioning), and *D* is the input dimension.
        timesteps: Per-token timestep embeddings, shape ``(B, T)``.
        positions: Per-token patch coordinates used to build the RoPE
            frequencies. With the default ``use_middle_indices_grid=True``,
            shape is ``(B, n_pos_dims, T, 2)`` where ``n_pos_dims=3`` for
            video (time, height, width) and ``n_pos_dims=1`` for audio
            (time); the last dim of size 2 holds the ``[start, end)``
            index bounds of each patch, and RoPE is evaluated at the
            *middle* of that range -- hence the flag name. Taking the
            patch midpoint produces a smoother and more accurate
            positional signal than indexing by the patch's start when
            patches span more than one spatial / temporal unit.
            When ``use_middle_indices_grid=False``, the legacy 3-D form
            ``(B, n_pos_dims, T)`` of integer positional indices is
            accepted instead and used as-is (no midpoint derivation).
        context: Text conditioning embeddings from the prompt encoder.
        enabled: Whether this modality is active in the current forward pass.
        context_mask: Optional mask for the text context tokens.
        attention_mask: Optional 2-D self-attention mask, shape ``(B, T, T)``.
            Values in ``[0, 1]`` where ``1`` = full attention and ``0`` = no
            attention. ``None`` means unrestricted (full) attention between
            all tokens. Built incrementally by conditioning items; see
            :class:`~ltx_core.conditioning.types.attention_strength_wrapper.ConditioningItemAttentionStrengthWrapper`.
        keyframes_mask: Optional per-token marker, shape ``(B, T, 1)`` -- the same layout as
            ``timesteps`` -- non-zero for tokens whose latent encodes a *single standalone pixel
            frame* rather than the usual multi-frame span. That is the target's first latent frame
            (the video encoder is causal) plus any generated keyframe slots. Selects the tokens
            that receive the model's learned keyframe absolute-position embedding. ``None`` means
            no token is marked, which is also the effective behaviour of every model built without
            ``use_keyframes_abs_pos_embedding``.
    """

    latent: (
        torch.Tensor
    )  # Shape: (B, T, D) where B is the batch size, T is the number of tokens, and D is input dimension
    sigma: torch.Tensor  # Shape: (B,). Current sigma value, used for cross-attention timestep calculation.
    timesteps: torch.Tensor  # Shape: (B, T) where T is the number of timesteps
    # Shape: (B, n_pos_dims, T, 2) by default (use_middle_indices_grid=True);
    # n_pos_dims=3 for video, 1 for audio; last dim holds [start, end) patch bounds.
    # Legacy form (B, n_pos_dims, T) when use_middle_indices_grid=False.
    positions: torch.Tensor
    context: torch.Tensor
    enabled: bool = True
    context_mask: torch.Tensor | None = None
    attention_mask: torch.Tensor | None = None
    keyframes_mask: torch.Tensor | None = None  # Shape: (B, T, 1), non-zero on single-pixel-frame latents
    # WAM: ``(B, T)`` boolean, True on tokens that are robot actions rather than video patches.
    # Action tokens are interleaved among the video tokens in timestamp order -- the exact layout
    # is the training strategy's to decide -- so they share the stream's timesteps, RoPE clock,
    # self-attention and sequence, and their ordering stays meaningful if attention is later made
    # causal. They enter through ``action_in`` instead of ``patchify_proj`` and leave
    # through ``action_out`` instead of ``proj_out``. Their rows of ``latent`` carry the action
    # VAE's channels in ``[..., :action_channels]``, zero-padded out to the video ``in_channels``
    # so the sequence remains a single tensor and every shape downstream is unchanged.
    # ``None`` -- the default -- means no action tokens, which is every non-WAM model.
    action_mask: torch.Tensor | None = None

    def split(self, sizes: list[int]) -> list[Modality]:
        """Split along the batch dimension into chunks of the given sizes."""
        n = len(sizes)
        split_fields: dict[str, list[torch.Tensor | None] | list[bool]] = {}
        for f in dataclasses.fields(self):
            value = getattr(self, f.name)
            if isinstance(value, torch.Tensor):
                split_fields[f.name] = list(value.split(sizes, dim=0))
            elif value is None or isinstance(value, bool):
                split_fields[f.name] = [value] * n
            else:
                raise TypeError(f"Cannot split field {f.name!r}: unsupported type {type(value)}")
        return [Modality(**{name: parts[i] for name, parts in split_fields.items()}) for i in range(n)]
