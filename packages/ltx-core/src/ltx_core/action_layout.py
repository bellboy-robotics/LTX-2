"""Where action tokens sit inside a video token sequence, and how to splice them in and out.

The WAM interleaves robot-action tokens into the video sequence rather than running them as a
third stream, so the layout -- which positions are actions, in what order, with what RoPE times --
is a contract between training and inference. Training builds a sequence in this shape; inference
must build the identical one, or the model reads its visual context from the wrong frame and the
decoded actions are wrong without anything raising.

This module is the single definition of that shape, held here in ``ltx_core`` beside the other
sequence-layout code (``VideoLatentTools`` builds positions and the keyframes mask; the
transformer consumes ``Modality.action_mask``) so both callers import it rather than each
describing the geometry for itself. It mirrors ``GeneratedKeyframeLayout``: a recorded layout plus
a check that refuses to read tokens out of a sequence it does not describe.

The geometry
------------
Action ``k`` is the transition from pixel frame ``k`` to ``k+1``. Its RoPE time is ``k / fps`` --
when the motion starts -- and the frame it *produces* is ``k+1``. Each action token is placed
immediately after the video tokens of the latent frame holding the pixel frame it produces:
grouping by destination, not by source.

The video encoder is causal, so latent frame 0 holds pixel frame 0 alone and latent frame
``f >= 1`` holds pixel frames ``S(f-1)+1 .. Sf`` for a temporal scale ``S`` of 8. Latent frame 0 is
the conditioning frame -- nothing was predicted to reach it -- so it carries no action, and every
later frame carries the ``S`` actions that produced its ``S`` pixel frames. A 25-frame window at
1280x704 lays out as::

    880 video | 880 video | 8 actions | 880 video | 8 | 880 video | 8

Under bidirectional attention the placement is cosmetic: RoPE carries the time and attention is
permutation-invariant. It matters only if attention is later made causal, which is why the order
follows the clock rather than appending at the tail.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import cached_property

import torch

from ltx_core.tools import VideoLatentTools
from ltx_core.types import LatentState


@dataclass(frozen=True)
class ActionTokenLayout:
    """The positions action tokens occupy in an interleaved video/action sequence.

    Attributes:
        num_actions: How many action tokens the sequence carries, one per action in the clip.
        num_frames: Latent frames in the video sequence.
        tokens_per_frame: Video tokens per latent frame, at the target resolution.
        temporal_scale: Pixel frames per latent frame after the first -- the VAE's temporal scale.
        fps: Frame rate of the clip. The action clock has to be the video's clock, or the two
            modalities disagree about when things happen.
    """

    num_actions: int
    num_frames: int
    tokens_per_frame: int
    temporal_scale: int
    fps: float

    def __post_init__(self) -> None:
        if min(self.num_actions, self.num_frames, self.tokens_per_frame, self.temporal_scale) < 1:
            raise ValueError(
                f"every count must be positive: num_actions={self.num_actions}, "
                f"num_frames={self.num_frames}, tokens_per_frame={self.tokens_per_frame}, "
                f"temporal_scale={self.temporal_scale}"
            )
        if self.fps <= 0:
            raise ValueError(f"fps must be positive, got {self.fps}")
        if self.groups[-1] >= self.num_frames:
            raise ValueError(
                f"{self.num_actions} actions at temporal scale {self.temporal_scale} need "
                f"{self.groups[-1] + 1} latent frames but the video has {self.num_frames}"
            )

    @cached_property
    def groups(self) -> tuple[int, ...]:
        """Destination latent frame of each action: the one holding the pixel frame it produces."""
        return tuple(k // self.temporal_scale + 1 for k in range(self.num_actions))

    @cached_property
    def counts(self) -> tuple[int, ...]:
        """How many actions follow each latent frame's video tokens, indexed by frame."""
        counts = [0] * self.num_frames
        for group in self.groups:
            counts[group] += 1
        return tuple(counts)

    @cached_property
    def action_indices(self) -> tuple[int, ...]:
        """Index of each action token in the interleaved sequence, in action order."""
        indices: list[int] = []
        cursor = 0
        for frame in range(self.num_frames):
            cursor += self.tokens_per_frame
            for _ in range(self.counts[frame]):
                indices.append(cursor)
                cursor += 1
        return tuple(indices)

    @property
    def video_tokens(self) -> int:
        return self.num_frames * self.tokens_per_frame

    @property
    def total_tokens(self) -> int:
        return self.video_tokens + self.num_actions

    def check(self, seq_len: int, *, interleaved: bool) -> None:
        """Refuse to work on a sequence this layout does not describe.

        The failure this guards against is silent: a sequence built to a different layout has the
        same dtype and, if the counts happen to agree, the same shape. Nothing would raise; the
        action tokens would simply sit against the wrong video frames and decode to plausible
        nonsense. So the length is checked at every boundary rather than trusted.

        Args:
            seq_len: Token count of the sequence in hand.
            interleaved: Whether that sequence already holds the action tokens.
        """
        expected = self.total_tokens if interleaved else self.video_tokens
        if seq_len != expected:
            kind = "an interleaved" if interleaved else "a video-only"
            raise ValueError(
                f"expected {kind} sequence of {expected} tokens "
                f"({self.num_frames} frames x {self.tokens_per_frame} tokens"
                f"{f' + {self.num_actions} actions' if interleaved else ''}), got {seq_len}. "
                "The layout was built for a different resolution or clip length."
            )

    def interleave(self, video: torch.Tensor, action: torch.Tensor, dim: int) -> torch.Tensor:
        """Splice the action rows into the video rows along ``dim``, in timestamp order.

        Both tensors are indexed along one axis only -- token order -- so the same call serves
        latents ``(B, T, C)``, per-token scalars ``(B, T)`` and positions ``(B, 3, T, 2)``, which
        is why the axis is a parameter rather than assumed.
        """
        self.check(video.shape[dim], interleaved=False)
        if action.shape[dim] != self.num_actions:
            raise ValueError(f"expected {self.num_actions} action rows along dim {dim}, got {action.shape[dim]}")

        parts: list[torch.Tensor] = []
        cursor = 0
        for frame in range(self.num_frames):
            start, end = frame * self.tokens_per_frame, (frame + 1) * self.tokens_per_frame
            parts.append(video.narrow(dim, start, end - start))
            count = self.counts[frame]
            if count:
                parts.append(action.narrow(dim, cursor, count))
                cursor += count
        return torch.cat(parts, dim=dim)

    def extract(self, sequence: torch.Tensor, dim: int) -> torch.Tensor:
        """Pull the action rows back out of an interleaved sequence, in action order.

        The inverse of :meth:`interleave`, and how validation reads the model's action prediction
        out of what it generated. A pure read: the sequence is not modified, so this can be called
        at any point after denoising without disturbing what follows.
        """
        self.check(sequence.shape[dim], interleaved=True)
        index = torch.as_tensor(self.action_indices, device=sequence.device)
        return sequence.index_select(dim, index)

    def remove(self, sequence: torch.Tensor, dim: int) -> torch.Tensor:
        """Drop the action rows, leaving the video tokens in grid order.

        The video decode path assumes the token sequence *is* the patchified video grid --
        ``clear_conditioning`` keeps a leading slice of it and ``unpatchify`` reshapes that into
        ``(B, C, frames, H, W)``. Interleaved action rows break both, and they break them quietly:
        the shapes still work out, the picture just comes back scrambled. So the rows come out
        before anything downstream looks at the sequence.
        """
        self.check(sequence.shape[dim], interleaved=True)
        index = torch.as_tensor(self.video_indices, device=sequence.device)
        return sequence.index_select(dim, index)

    @cached_property
    def video_indices(self) -> tuple[int, ...]:
        """Index of each video token in the interleaved sequence, in grid order."""
        actions = set(self.action_indices)
        return tuple(i for i in range(self.total_tokens) if i not in actions)

    def mask(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """``(B, T)`` boolean over the interleaved sequence, true at the action tokens.

        This is what the transformer reads as ``Modality.action_mask`` to route those positions
        through ``action_in`` and ``action_out`` instead of the video projections.
        """
        flags = torch.zeros(batch_size, self.total_tokens, dtype=torch.bool, device=device)
        flags[:, torch.as_tensor(self.action_indices, device=device)] = True
        return flags

    def positions(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """``(B, 3, num_actions, 2)`` RoPE positions for the action tokens alone.

        Time is exactly ``k / fps``; actions have no spatial extent, so the height and width axes
        stay zero. The last axis holds ``[start, end)`` bounds and RoPE reads their midpoint, so
        leaving the bounds zero-width puts the token at ``k / fps`` rather than half a frame later.
        """
        times = torch.arange(self.num_actions, device=device, dtype=torch.float32) / self.fps
        positions = torch.zeros(batch_size, 3, self.num_actions, 2, device=device, dtype=torch.float32)
        positions[:, 0] = times.view(1, self.num_actions, 1).expand(batch_size, self.num_actions, 2)
        return positions

    def pad_channels(self, action_latents: torch.Tensor, channels: int) -> torch.Tensor:
        """Zero-pad ``(B, n, action_channels)`` out to the video channel count.

        The sequence is one tensor, so action rows have to be as wide as video rows. The padding
        is inert: it is noised and denoised along with everything else, and both the training loss
        and :func:`decode` slice it back off using the action VAE's channel count.
        """
        if action_latents.shape[1] != self.num_actions:
            raise ValueError(f"expected {self.num_actions} action rows, got {action_latents.shape[1]}")
        if action_latents.shape[2] > channels:
            raise ValueError(
                f"action latents have {action_latents.shape[2]} channels, more than the video's {channels}"
            )
        padded = torch.zeros(
            action_latents.shape[0],
            self.num_actions,
            channels,
            device=action_latents.device,
            dtype=action_latents.dtype,
        )
        padded[..., : action_latents.shape[2]] = action_latents
        return padded

    @classmethod
    def from_sequence(
        cls,
        *,
        num_actions: int,
        num_frames: int,
        video_seq_len: int,
        temporal_scale: int,
        fps: float,
    ) -> "ActionTokenLayout":
        """Build a layout from a video sequence whose per-frame token count is implied by its length."""
        if video_seq_len % num_frames:
            raise ValueError(f"video sequence length {video_seq_len} is not divisible by {num_frames} latent frames")
        return cls(
            num_actions=num_actions,
            num_frames=num_frames,
            tokens_per_frame=video_seq_len // num_frames,
            temporal_scale=temporal_scale,
            fps=fps,
        )


def add_action_slots(state: LatentState, layout: ActionTokenLayout) -> LatentState:
    """Splice empty action slots into a video ``LatentState``, ready to be generated.

    The inference-side counterpart to ``_interleave_action_tokens`` in the trainer. Both put the
    action rows in the places :class:`ActionTokenLayout` names; they differ only in what they
    splice into -- loose tensors there, a ``LatentState`` here -- because LTX itself keeps those
    two representations apart.

    Call this after the conditioning items have been applied and *before* the noiser. The slots
    go in holding zeros with ``denoise_mask = 1``, which is what tells ``GaussianNoiser`` to fill
    them with noise and the denoising loop to generate them: exactly the treatment the unconditioned
    video tokens get, no special case anywhere. ``keyframes_mask`` is zero for them -- that marker
    is about latents encoding a single pixel frame, which an action is not.

    Args:
        state: The video state, still video-only.
        layout: Where the action rows go. Usually the one carried by ``WamVideoLatentTools``.

    Returns:
        The same state with ``num_actions`` rows spliced into ``latent``, ``clean_latent``,
        ``denoise_mask``, ``positions`` and ``keyframes_mask``.
    """
    batch_size = state.latent.shape[0]
    device, dtype = state.latent.device, state.latent.dtype
    channels = state.latent.shape[-1]

    empty = torch.zeros(batch_size, layout.num_actions, channels, device=device, dtype=dtype)
    generate = torch.ones(
        batch_size, layout.num_actions, 1, device=device, dtype=state.denoise_mask.dtype
    )

    return replace(
        state,
        latent=layout.interleave(state.latent, empty, dim=1),
        clean_latent=layout.interleave(state.clean_latent, empty, dim=1),
        denoise_mask=layout.interleave(state.denoise_mask, generate, dim=1),
        positions=layout.interleave(state.positions, layout.positions(batch_size, device), dim=2),
        keyframes_mask=(
            None
            if state.keyframes_mask is None
            else layout.interleave(
                state.keyframes_mask,
                torch.zeros(
                    batch_size, layout.num_actions, 1, device=device, dtype=state.keyframes_mask.dtype
                ),
                dim=1,
            )
        ),
    )


@dataclass(frozen=True)
class WamVideoLatentTools(VideoLatentTools):
    """``VideoLatentTools`` that knows the sequence carries action tokens.

    Only one behaviour changes: :meth:`clear_conditioning` drops the action rows before handing
    the sequence on. Everything downstream of it -- ``unpatchify``, the VAE decode, the tiling --
    reads the token sequence as the patchified video grid, so the rows have to be gone by then.

    The override rather than a call at each site is deliberate: forgetting it does not raise, it
    silently decodes a scrambled video. Putting it here means no caller can forget.

    The layout lives on the tools because that is already where the sequence's shape lives --
    ``target_shape``, ``tokens_per_latent_frame``, the patchifier. Reading the actions back out
    needs no state at all: :meth:`ActionTokenLayout.extract` is a pure read the caller does
    directly, whenever it likes, before finalizing.
    """

    action_layout: ActionTokenLayout | None = None

    def clear_conditioning(self, latent_state: LatentState) -> LatentState:
        if self.action_layout is None:
            return super().clear_conditioning(latent_state)
        return super().clear_conditioning(
            replace(
                latent_state,
                latent=self.action_layout.remove(latent_state.latent, dim=1),
                clean_latent=self.action_layout.remove(latent_state.clean_latent, dim=1),
                denoise_mask=self.action_layout.remove(latent_state.denoise_mask, dim=1),
                positions=self.action_layout.remove(latent_state.positions, dim=2),
                keyframes_mask=(
                    None
                    if latent_state.keyframes_mask is None
                    else self.action_layout.remove(latent_state.keyframes_mask, dim=1)
                ),
            )
        )
