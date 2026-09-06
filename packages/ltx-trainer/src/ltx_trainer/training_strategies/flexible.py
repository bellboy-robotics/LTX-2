"""Flexible training strategy for a unified conditioning framework.
This strategy implements the Unified Conditioning Framework that supports:
- Simple fine-tuning with text conditioning (text-to-video/audio)
- Intrinsic conditioning (first_frame, prefix, suffix, spatial_crop, mask)
- Extrinsic conditioning (concatenation-based, IC-LoRA style)
The flexible strategy replaces TextToVideoStrategy and VideoToVideoStrategy by expressing
all conditioning scenarios through configuration rather than code.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal, Union

import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator
from torch import Tensor

from ltx_core.model.transformer.modality import Modality
from ltx_core.types import SpatioTemporalScaleFactors
from ltx_trainer.timestep_samplers import TimestepSampler
from ltx_trainer.training_strategies.base_strategy import (
    DEFAULT_FPS,
    ModelInputs,
    TrainingStrategy,
    TrainingStrategyConfigBase,
)

# =============================================================================
# Configuration Classes
# =============================================================================


class IntrinsicConditionBase(BaseModel):
    """Base for intrinsic conditioning — tokens get clean latents, timestep=0, no loss."""

    model_config = ConfigDict(extra="forbid")

    probability: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Probability of applying this condition",
    )


class FirstFrameConditionConfig(IntrinsicConditionBase):
    """First frame conditioning — frame 0 is clean, excluded from loss."""

    type: Literal["first_frame"] = "first_frame"


class PrefixConditionConfig(IntrinsicConditionBase):
    """Prefix conditioning — first N temporal units are clean, excluded from loss."""

    type: Literal["prefix"] = "prefix"
    temporal_boundary: int = Field(
        ...,
        ge=1,
        description="Number of temporal units for prefix region. "
        "For video: number of latent frames. For audio: number of audio latent timesteps.",
    )


class SuffixConditionConfig(IntrinsicConditionBase):
    """Suffix conditioning — last N temporal units are clean, excluded from loss."""

    type: Literal["suffix"] = "suffix"
    temporal_boundary: int = Field(
        ...,
        ge=1,
        description="Number of temporal units for suffix region. "
        "For video: number of latent frames. For audio: number of audio latent timesteps.",
    )


class SpatialCropConditionConfig(IntrinsicConditionBase):
    """Spatial crop conditioning — rectangular pixel region is clean, excluded from loss."""

    type: Literal["spatial_crop"] = "spatial_crop"
    spatial_region: tuple[int, int, int, int] = Field(
        ...,
        description="Spatial crop region as (y1, x1, y2, x2) in pixel coordinates",
    )


class MaskConditionConfig(IntrinsicConditionBase):
    """Mask conditioning — per-sample binary mask determines conditioning tokens."""

    type: Literal["mask"] = "mask"
    mask_dir: str = Field(
        ...,
        description="Directory containing per-sample masks",
    )


class ReferenceConditionConfig(BaseModel):
    """Reference conditioning (IC-LoRA style concatenation).
    External reference latents are concatenated to the target sequence.
    Reference tokens are clean (timestep=0), excluded from loss, and
    participate in bidirectional self-attention.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["reference"] = "reference"
    latents_dir: str = Field(..., description="Directory for reference latents")
    probability: float = Field(default=1.0, ge=0.0, le=1.0, description="Probability of applying this condition")


# Discriminated union for condition configs
ConditionConfig = Annotated[
    Union[
        FirstFrameConditionConfig,
        PrefixConditionConfig,
        SuffixConditionConfig,
        SpatialCropConditionConfig,
        MaskConditionConfig,
        ReferenceConditionConfig,
    ],
    Field(discriminator="type"),
]


class ModalityConfig(BaseModel):
    """Configuration for a single modality (video or audio)."""

    model_config = ConfigDict(extra="forbid")

    is_generated: bool = Field(
        ...,
        description="True = generated modality (denoised, contributes to loss), False = conditioning-only modality",
    )

    latents_dir: str = Field(
        ...,
        description="Directory for latents",
    )

    conditions: list[ConditionConfig] = Field(
        default_factory=list,
        description="List of conditions (e.g. first_frame, prefix, reference). Text conditioning is always applied.",
    )


class ActionConfig(BaseModel):
    """Configuration for the WAM robot-action stream.

    Actions are not a third modality. One action token per 15 Hz step is interleaved into the
    video sequence in timestamp order, sharing its noise level, RoPE clock and self-attention.
    They enter and leave the transformer through ``action_in`` / ``action_out`` rather than
    ``patchify_proj`` / ``proj_out``.
    """

    model_config = ConfigDict(extra="forbid")

    latents_dir: str = Field(
        ...,
        description="Directory holding the action VAE latents, one (num_actions, action_channels) array per clip.",
    )

    loss_weight: float = Field(
        5.0,
        gt=0,
        description="Lambda on the action loss term. Action tokens are ~0.7% of the sequence, so "
        "an unweighted sum buries them. Swept 1 / 5 / 10 / 20.",
    )


class FlexibleStrategyConfig(TrainingStrategyConfigBase):
    """Configuration for the flexible training strategy.
    This strategy supports all conditioning scenarios through configuration:
    - Text-to-video/audio with simple fine-tuning
    - Intrinsic conditioning like first-frame, extension, outpainting
    - Reference conditioning like IC-LoRA (concatenation-based reference)
    """

    name: Literal["flexible"] = "flexible"

    video: ModalityConfig | None = Field(
        default=None,
        description="Video modality configuration",
    )

    audio: ModalityConfig | None = Field(
        default=None,
        description="Audio modality configuration",
    )

    action: ActionConfig | None = Field(
        default=None,
        description="WAM action stream, interleaved into the video sequence. None = no actions.",
    )

    @model_validator(mode="after")
    def validate_action_requires_generated_video(self) -> "FlexibleStrategyConfig":
        """Actions ride inside the video sequence, so that sequence has to exist and be denoised."""
        if self.action is None:
            return self
        if self.video is None or not self.video.is_generated:
            raise ValueError("action requires a video modality with is_generated=true")
        return self

    @model_validator(mode="after")
    def validate_at_least_one_generated(self) -> "FlexibleStrategyConfig":
        """Ensure at least one modality has is_generated=true."""
        has_video_target = self.video is not None and self.video.is_generated
        has_audio_target = self.audio is not None and self.audio.is_generated
        if not has_video_target and not has_audio_target:
            raise ValueError("At least one modality must have is_generated=true")
        return self

    @model_validator(mode="after")
    def validate_audio_intrinsic_regions(self) -> "FlexibleStrategyConfig":
        """Reject video-only intrinsic regions on the audio modality."""
        if self.audio is None:
            return self
        for cond in self.audio.conditions:
            if isinstance(cond, (FirstFrameConditionConfig, SpatialCropConditionConfig)):
                raise ValueError(
                    f"Intrinsic condition '{cond.type}' is not supported for audio. "
                    f"Audio supports: prefix, suffix, mask."
                )
        return self

    def get_data_sources(self) -> dict[str, str]:
        """Dynamically determine required data sources from config.
        Returns a mapping of directory name (under ``preprocessed_data_root``) to
        the dataset output key.
        """
        sources: dict[str, str] = {"conditions": "conditions"}

        if self.video is not None:
            sources[self.video.latents_dir] = "video_latents"
        if self.audio is not None:
            sources[self.audio.latents_dir] = "audio_latents"
        if self.action is not None:
            sources[self.action.latents_dir] = "action_latents"

        for modality_config in (self.video, self.audio):
            if modality_config is None:
                continue
            for cond in modality_config.conditions:
                if isinstance(cond, ReferenceConditionConfig):
                    sources[cond.latents_dir] = cond.latents_dir
                elif isinstance(cond, MaskConditionConfig):
                    sources[cond.mask_dir] = cond.mask_dir

        return sources


# =============================================================================
# Helper Data Structures
# =============================================================================


@dataclass
class ModalityProcessingResult:
    """Result of processing a single modality."""

    modality: Modality
    targets: Tensor | None
    loss_mask: Tensor | None
    # WAM: (B, T) True on the interleaved action tokens. None when there are none.
    action_mask: Tensor | None = None


@dataclass
class LatentData:
    """Loaded and patchified latents with metadata."""

    latents: Tensor  # [B, seq_len, C]
    num_frames: int
    height: int
    width: int
    fps: float


# =============================================================================
# FlexibleStrategy Implementation
# =============================================================================


class FlexibleStrategy(TrainingStrategy):
    """Unified training strategy supporting all conditioning scenarios.
    This strategy implements the Unified Conditioning Framework, allowing
    any training scenario to be expressed through configuration.
    """

    config: FlexibleStrategyConfig

    def __init__(self, config: FlexibleStrategyConfig):
        """Initialize strategy with configuration.
        Args:
            config: Flexible strategy configuration
        """
        super().__init__(config)
        self.config = config
        self.reference_spatial_scale_factor, self.reference_temporal_scale_factor = (
            self._infer_reference_scale_factors_from_config()
        )

    def prepare_training_inputs(
        self,
        batch: dict[str, Any],
        timestep_sampler: TimestepSampler,
    ) -> ModelInputs:
        """Prepare training inputs by processing video and audio modalities."""
        video_result = self._process_modality(self.config.video, batch, "video", timestep_sampler)
        audio_result = self._process_modality(self.config.audio, batch, "audio", timestep_sampler)

        return ModelInputs(
            video=video_result.modality if video_result else None,
            audio=audio_result.modality if audio_result else None,
            video_targets=video_result.targets if video_result else None,
            audio_targets=audio_result.targets if audio_result else None,
            video_loss_mask=video_result.loss_mask if video_result else None,
            audio_loss_mask=audio_result.loss_mask if audio_result else None,
            video_action_mask=video_result.action_mask if video_result else None,
            action_loss_weight=self.config.action.loss_weight if self.config.action else 1.0,
            action_channels=batch["action_latents"].shape[-1] if self.config.action else 0,
        )

    def compute_loss(
        self,
        video_pred: Tensor | None,
        audio_pred: Tensor | None,
        inputs: ModelInputs,
    ) -> Tensor:
        """Compute masked MSE loss for video and audio predictions. Returns [B,].

        With the WAM action stream the video sequence carries two kinds of token, so the video
        term splits in two: ``video_mse + lambda * action_mse``. The action term is restricted to
        the action tokens *and* to the leading ``action_channels``, since the rest of an action
        row is zero padding. Both components are recorded on ``self.last_loss_components`` so the
        trainer can log them separately -- the plan calls for that from the start, since lambda
        is unreadable otherwise.
        """
        total_loss = None
        self.last_loss_components: dict[str, Tensor] = {}

        if video_pred is not None and inputs.video_targets is not None:
            action_mask = inputs.video_action_mask
            if action_mask is None:
                video_loss = self._compute_modality_loss(
                    pred=video_pred,
                    targets=inputs.video_targets,
                    loss_mask=inputs.video_loss_mask,
                )
            else:
                channels = inputs.action_channels
                video_loss = self._compute_modality_loss(
                    pred=video_pred,
                    targets=inputs.video_targets,
                    loss_mask=inputs.video_loss_mask & ~action_mask,
                )
                action_loss = self._compute_modality_loss(
                    pred=video_pred[..., :channels],
                    targets=inputs.video_targets[..., :channels],
                    loss_mask=inputs.video_loss_mask & action_mask,
                )
                self.last_loss_components["video_loss"] = video_loss.detach()
                self.last_loss_components["action_loss"] = action_loss.detach()
                video_loss = video_loss + inputs.action_loss_weight * action_loss
            total_loss = video_loss

        if audio_pred is not None and inputs.audio_targets is not None:
            audio_loss = self._compute_modality_loss(
                pred=audio_pred,
                targets=inputs.audio_targets,
                loss_mask=inputs.audio_loss_mask,
            )
            total_loss = audio_loss if total_loss is None else total_loss + audio_loss

        if total_loss is None:
            raise ValueError("No valid predictions and targets provided for loss computation")

        return total_loss

    def get_checkpoint_metadata(self) -> dict[str, Any]:
        """Include reference scale factors in checkpoint metadata for inference pipelines."""
        metadata: dict[str, Any] = {}
        spatial = self.reference_spatial_scale_factor
        temporal = self.reference_temporal_scale_factor
        if spatial is not None and spatial != 1:
            metadata["reference_spatial_scale_factor"] = spatial
            metadata["reference_downscale_factor"] = spatial  # backward compat
        if temporal is not None and temporal != 1:
            metadata["reference_temporal_scale_factor"] = temporal
        return metadata

    def _infer_reference_scale_factors_from_config(self) -> tuple[int | None, int | None]:
        """Infer spatial and temporal scale factors by peeking at one sample pair."""
        if self.config.video is None:
            return None, None
        for cond in self.config.video.conditions:
            if not isinstance(cond, ReferenceConditionConfig):
                continue
            target_dir = Path(self.config.video.latents_dir)
            ref_dir = Path(cond.latents_dir)
            for sample_file in target_dir.rglob("*.pt"):
                ref_file = ref_dir / sample_file.relative_to(target_dir)
                if not ref_file.exists():
                    continue
                target_data = torch.load(sample_file, map_location="cpu", weights_only=True)
                ref_data = torch.load(ref_file, map_location="cpu", weights_only=True)
                if "height" not in ref_data or "height" not in target_data:
                    continue
                spatial = self._infer_scale_factor(
                    ref_data["height"],
                    ref_data["width"],
                    target_data["height"],
                    target_data["width"],
                )
                temporal = self._infer_temporal_scale_factor(
                    ref_data["num_frames"],
                    target_data["num_frames"],
                )
                return spatial, temporal
        return None, None

    def _process_modality(
        self,
        modality_config: ModalityConfig | None,
        batch: dict[str, Any],
        modality_key: str,
        timestep_sampler: TimestepSampler,
    ) -> ModalityProcessingResult | None:
        """Process a single modality: load latents, add noise, apply conditions, build Modality."""
        if modality_config is None:
            return None

        # Step 1: Load and patchify latents
        data = self._patchify_latent_data(batch[f"{modality_key}_latents"], modality_key)
        latents = data.latents

        batch_size, seq_len, _ = latents.shape
        device = latents.device
        dtype = latents.dtype

        # Step 2: Get text embeddings
        conditions = batch["conditions"]
        prompt_embeds = conditions[f"{modality_key}_prompt_embeds"]
        prompt_attention_mask = conditions["prompt_attention_mask"]

        # Step 3: Initialize noise, timesteps, and loss mask based on is_generated flag
        if modality_config.is_generated:
            noisy_latents, targets, timesteps, loss_mask, sigmas = self._initialize_noisy_target(
                latents, timestep_sampler
            )
        else:
            # Conditioning modality: keep clean (sigma=0), no loss
            noisy_latents = latents
            targets = None
            timesteps = torch.zeros(batch_size, seq_len, device=device, dtype=dtype)
            loss_mask = None
            sigmas = torch.zeros(batch_size, device=device, dtype=dtype)

        # Step 4: Generate positions
        if modality_key == "video":
            positions = self._get_video_positions(
                num_frames=data.num_frames,
                height=data.height,
                width=data.width,
                batch_size=batch_size,
                fps=data.fps,
                device=device,
            )
        else:
            positions = self._get_audio_positions(
                num_time_steps=seq_len,
                batch_size=batch_size,
                device=device,
            )

        # Step 5: Apply conditions (intrinsic first, then extrinsic)
        for cond in modality_config.conditions:
            if isinstance(cond, IntrinsicConditionBase) and modality_config.is_generated:
                noisy_latents, timesteps, loss_mask = self._apply_intrinsic_condition(
                    noisy_latents=noisy_latents,
                    clean_latents=latents,
                    timesteps=timesteps,
                    loss_mask=loss_mask,
                    config=cond,
                    height=data.height,
                    width=data.width,
                    batch=batch,
                )

        for cond in modality_config.conditions:
            if isinstance(cond, ReferenceConditionConfig):
                noisy_latents, positions, timesteps, loss_mask, targets = self._apply_reference_condition(
                    noisy_latents=noisy_latents,
                    positions=positions,
                    timesteps=timesteps,
                    loss_mask=loss_mask,
                    targets=targets,
                    batch=batch,
                    config=cond,
                    modality_key=modality_key,
                )

        # Step 5b: Interleave the WAM action tokens into the video sequence
        action_mask = None
        if modality_key == "video" and self.config.action is not None:
            noisy_latents, targets, timesteps, loss_mask, positions, action_mask = self._interleave_action_tokens(
                action_latents=batch["action_latents"],
                noisy_latents=noisy_latents,
                targets=targets,
                timesteps=timesteps,
                loss_mask=loss_mask,
                positions=positions,
                sigmas=sigmas,
                num_frames=data.num_frames,
                fps=data.fps,
            )

        # Step 6: Build Modality
        modality = Modality(
            enabled=True,
            latent=noisy_latents,
            sigma=sigmas,
            timesteps=timesteps,
            positions=positions,
            context=prompt_embeds,
            context_mask=prompt_attention_mask,
            action_mask=action_mask,
        )

        return ModalityProcessingResult(
            modality=modality,
            targets=targets,
            loss_mask=loss_mask,
            action_mask=action_mask,
        )

    def _interleave_action_tokens(  # noqa: PLR0913
        self,
        action_latents: Tensor,
        noisy_latents: Tensor,
        targets: Tensor,
        timesteps: Tensor,
        loss_mask: Tensor,
        positions: Tensor,
        sigmas: Tensor,
        num_frames: int,
        fps: float,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Splice the action tokens into the video sequence in timestamp order.

        Action ``k`` is the transition from pixel frame ``k`` to ``k+1``, so its RoPE time is
        ``k / fps`` and the frame it *produces* is ``k+1``. Each action token is placed
        immediately after the video tokens of the latent frame holding the pixel frame it
        produces -- grouping by destination rather than by source, matching how the windows were
        cut in ``build_ltx_manifest.py``.

        With the causal VAE, latent frame 0 holds pixel frame 0 alone and latent frame ``f >= 1``
        holds pixels ``S(f-1)+1 .. Sf`` for a temporal scale ``S`` of 8. So latent frame 0 -- the
        conditioning frame, clean and excluded from the loss -- carries no action, nothing having
        been predicted for it, and every later frame carries the ``S`` actions that produced its
        ``S`` pixel frames. A 25-frame window lays out as::

            880 video | 880 video | 8 actions | 880 video | 8 | 880 video | 8

        Under bidirectional attention the placement is cosmetic -- RoPE carries the time and
        attention is permutation-invariant. It matters only if attention is later made causal,
        which is why the order follows the clock rather than appending at the tail.

        Action latents are zero-padded from ``action_channels`` out to the video channel count so
        the sequence stays a single tensor. They take the *same* sigma as the video tokens
        (shared noise level, per the plan), and the padded channels are noised along with the
        rest -- the loss ignores them via ``ModelInputs.action_channels``.

        Args:
            action_latents: ``(B, n_actions, action_channels)`` from the frozen action VAE.
            noisy_latents / targets / timesteps / loss_mask / positions: the video sequence.
            sigmas: ``(B,)`` noise level, reused for the action tokens.
            num_frames: number of latent frames in the video sequence.
            fps: the video's frame rate; the action clock has to be the same one.

        Returns:
            ``(noisy_latents, targets, timesteps, loss_mask, positions, action_mask)`` -- the five
            video tensors with the action rows spliced in at their timestamp positions, so ``T``
            grows from 3520 to 3544, plus the new ``(B, T)`` boolean marking which rows those are.

        Raises:
            ValueError: if the video modality is conditioning-only, which leaves ``targets`` and
                ``loss_mask`` unset. ``FlexibleStrategyConfig`` already rejects that combination;
                this is the same requirement stated where the code depends on it.
        """
        if targets is None or loss_mask is None:
            raise ValueError(
                "action interleaving requires a generated video modality: a conditioning-only one "
                "has no targets or loss mask to splice into"
            )
        batch_size, seq_len, channels = noisy_latents.shape
        num_actions = action_latents.shape[1]
        action_channels = action_latents.shape[2]
        device, dtype = noisy_latents.device, noisy_latents.dtype

        if seq_len % num_frames:
            raise ValueError(f"video sequence length {seq_len} is not divisible by {num_frames} latent frames")
        tokens_per_frame = seq_len // num_frames
        temporal_scale = int(self.video_scale_factors.time)

        # Pad the action latents out to the video channel count.
        padded = torch.zeros(batch_size, num_actions, channels, device=device, dtype=dtype)
        padded[..., :action_channels] = action_latents.to(device=device, dtype=dtype)

        # Same noise level as the video tokens, and the same velocity target.
        noise = torch.randn_like(padded)
        sigmas_expanded = sigmas.view(-1, 1, 1)
        noisy_actions = (1 - sigmas_expanded) * padded + sigmas_expanded * noise
        action_targets = noise - padded
        action_timesteps = sigmas.view(-1, 1).expand(batch_size, num_actions).clone()
        action_loss_mask = torch.ones(batch_size, num_actions, dtype=torch.bool, device=device)

        # Positions: time is exactly k / fps, and actions have no spatial extent. The last axis
        # holds [start, end) bounds and RoPE reads their midpoint, so zero-width bounds put the
        # token at k / fps rather than half a frame later.
        times = (torch.arange(num_actions, device=device, dtype=torch.float32) / fps).view(1, 1, num_actions, 1)
        action_positions = torch.zeros(batch_size, 3, num_actions, 2, device=device, dtype=torch.float32)
        action_positions[:, 0] = times.expand(batch_size, 1, num_actions, 2)[:, 0]

        # Which latent frame each action belongs to: the one holding the pixel frame it produces.
        groups = [k // temporal_scale + 1 for k in range(num_actions)]
        if groups and groups[-1] >= num_frames:
            raise ValueError(
                f"{num_actions} actions at temporal scale {temporal_scale} need "
                f"{groups[-1] + 1} latent frames but the video has {num_frames}"
            )

        latent_parts, target_parts, timestep_parts, mask_parts, position_parts, flag_parts = [], [], [], [], [], []
        cursor = 0
        for frame in range(num_frames):
            start, end = frame * tokens_per_frame, (frame + 1) * tokens_per_frame
            latent_parts.append(noisy_latents[:, start:end])
            timestep_parts.append(timesteps[:, start:end])
            position_parts.append(positions[:, :, start:end])
            flag_parts.append(torch.zeros(batch_size, tokens_per_frame, dtype=torch.bool, device=device))
            target_parts.append(targets[:, start:end])
            mask_parts.append(loss_mask[:, start:end])

            count = sum(1 for g in groups if g == frame)
            if not count:
                continue
            stop = cursor + count
            latent_parts.append(noisy_actions[:, cursor:stop])
            timestep_parts.append(action_timesteps[:, cursor:stop])
            position_parts.append(action_positions[:, :, cursor:stop])
            flag_parts.append(torch.ones(batch_size, count, dtype=torch.bool, device=device))
            target_parts.append(action_targets[:, cursor:stop])
            mask_parts.append(action_loss_mask[:, cursor:stop])
            cursor = stop

        return (
            torch.cat(latent_parts, dim=1),
            torch.cat(target_parts, dim=1),
            torch.cat(timestep_parts, dim=1),
            torch.cat(mask_parts, dim=1),
            torch.cat(position_parts, dim=2),
            torch.cat(flag_parts, dim=1),
        )

    @staticmethod
    def _initialize_noisy_target(
        latents: Tensor,
        timestep_sampler: TimestepSampler,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Add noise to latents and create training targets. Returns (noisy, targets, timesteps, mask, sigmas)."""
        batch_size, seq_len, _ = latents.shape
        sigmas = timestep_sampler.sample_for(latents)
        noise = torch.randn_like(latents)
        sigmas_expanded = sigmas.view(-1, 1, 1)
        noisy_latents = (1 - sigmas_expanded) * latents + sigmas_expanded * noise
        targets = noise - latents  # velocity prediction
        timesteps = sigmas.view(-1, 1).expand(batch_size, seq_len).clone()
        loss_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=latents.device)
        return noisy_latents, targets, timesteps, loss_mask, sigmas

    def _apply_intrinsic_condition(
        self,
        noisy_latents: Tensor,
        clean_latents: Tensor,
        timesteps: Tensor,
        loss_mask: Tensor,
        config: IntrinsicConditionBase,
        height: int,
        width: int,
        batch: dict[str, Any],
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Apply intrinsic conditioning using a binary mask.
        For each token, the mask value determines conditioning strength:
        - mask=1: conditioned (clean latent, timestep=0, excluded from loss)
        - mask=0: generated (noisy latent, original timestep, contributes to loss)
        The conditioning decision is drawn independently per batch element so the training
        signal across samples in a batch is i.i.d. -- a single batch-wide draw would
        correlate gradient updates across the batch.
        """
        batch_size, seq_len, _ = noisy_latents.shape
        device = noisy_latents.device

        # Per-sample Bernoulli draw -- each element is independently conditioned.
        apply_per_sample = torch.rand(batch_size, device=device) < config.probability
        if not apply_per_sample.any():
            return noisy_latents, timesteps, loss_mask

        if isinstance(config, FirstFrameConditionConfig):
            mask = self._compute_temporal_mask(batch_size, seq_len, height, width, 1, False, device)
        elif isinstance(config, PrefixConditionConfig):
            mask = self._compute_temporal_mask(
                batch_size, seq_len, height, width, config.temporal_boundary, False, device
            )
        elif isinstance(config, SuffixConditionConfig):
            mask = self._compute_temporal_mask(
                batch_size, seq_len, height, width, config.temporal_boundary, True, device
            )
        elif isinstance(config, SpatialCropConditionConfig):
            mask = self._compute_spatial_crop_mask(
                batch_size, seq_len, height, width, config.spatial_region, device, self.video_scale_factors
            )
        elif isinstance(config, MaskConditionConfig):
            # Binarize to match inference, which thresholds masks at load time
            # (validation_runner._load_and_downsample_mask / _load_audio_mask).
            mask = (batch[config.mask_dir]["mask"].reshape(batch_size, seq_len) > 0.5).float()
        else:
            raise ValueError(f"Unknown intrinsic condition type: {type(config).__name__}")

        # Zero the mask for samples the per-sample draw did not select.
        mask = mask * apply_per_sample.view(-1, 1).to(mask.dtype)

        # Apply binary mask: clean conditioned tokens, noisy generated tokens.
        m = mask.unsqueeze(-1)
        noisy_latents = m * clean_latents + (1 - m) * noisy_latents
        timesteps = (1 - mask) * timesteps
        loss_mask = loss_mask & (mask == 0)

        return noisy_latents, timesteps, loss_mask

    @staticmethod
    def _compute_temporal_mask(
        batch_size: int,
        seq_len: int,
        height: int,
        width: int,
        num_frames: int,
        from_end: bool,
        device: torch.device,
    ) -> Tensor:
        """Compute float mask for temporal region (prefix or suffix). Returns [B, seq_len] in {0, 1}."""
        tokens_per_frame = height * width
        num_tokens = num_frames * tokens_per_frame
        mask = torch.zeros(batch_size, seq_len, device=device)
        if from_end:
            mask[:, -num_tokens:] = 1.0
        else:
            mask[:, :num_tokens] = 1.0
        return mask

    @staticmethod
    def _compute_spatial_crop_mask(
        batch_size: int,
        seq_len: int,
        height: int,
        width: int,
        region: tuple[int, int, int, int],
        device: torch.device,
        scale_factors: SpatioTemporalScaleFactors,
    ) -> Tensor:
        """Compute float mask for spatial crop region (y1, x1, y2, x2) in pixel coords.
        Returns [B, seq_len] in {0, 1}.
        """
        y1, x1, y2, x2 = region
        num_frames = seq_len // (height * width)

        # Convert pixel to latent coordinates and clamp (per-axis VAE scale factor).
        def to_latent(v: int, scale: int, max_v: int) -> int:
            return max(0, min(v // scale, max_v))

        ly1 = to_latent(y1, scale_factors.height, height)
        ly2 = to_latent(y2, scale_factors.height, height)
        lx1 = to_latent(x1, scale_factors.width, width)
        lx2 = to_latent(x2, scale_factors.width, width)

        # Create spatial mask and tile across frames
        spatial_mask = torch.zeros(height, width, device=device)
        spatial_mask[ly1:ly2, lx1:lx2] = 1.0
        full_mask = spatial_mask.flatten().repeat(num_frames)

        return full_mask.unsqueeze(0).expand(batch_size, -1)

    def _patchify_latent_data(self, latent_data: dict[str, Any], modality_key: str) -> LatentData:
        """Patchify latent data and extract metadata."""
        latents = latent_data["latents"]

        if modality_key == "video":
            num_frames = latent_data["num_frames"][0].item()
            height = latent_data["height"][0].item()
            width = latent_data["width"][0].item()
            fps = latent_data.get("fps")
            fps = fps[0].item() if fps is not None else DEFAULT_FPS
            latents = self._video_patchifier.patchify(latents)
        else:
            num_frames = latent_data.get("num_frames", [latents.shape[2]])[0]
            if isinstance(num_frames, Tensor):
                num_frames = num_frames.item()
            height = 1
            width = 1
            fps = 1.0
            latents = self._audio_patchifier.patchify(latents)

        return LatentData(latents=latents, num_frames=num_frames, height=height, width=width, fps=fps)

    def _apply_reference_condition(
        self,
        noisy_latents: Tensor,
        positions: Tensor,
        timesteps: Tensor,
        loss_mask: Tensor | None,
        targets: Tensor | None,
        batch: dict[str, Any],
        config: ReferenceConditionConfig,
        modality_key: str,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor | None, Tensor | None]:
        """Concatenate reference latents to target sequence for reference conditioning (IC-LoRA style).
        The apply/skip decision is batch-wide (reference conditioning changes the sequence
        length, so it cannot be applied to only part of a batch) but is drawn from the torch
        RNG so runs are reproducible under ``torch.manual_seed`` — mirroring the intrinsic
        per-sample draw rather than Python's unseeded ``random``.
        """
        if torch.rand((), device=noisy_latents.device).item() >= config.probability:
            return noisy_latents, positions, timesteps, loss_mask, targets

        # Load and patchify condition latents
        cond = self._patchify_latent_data(batch[config.latents_dir], modality_key)
        cond_latents = cond.latents

        batch_size, cond_seq_len, _ = cond_latents.shape
        device = cond_latents.device
        dtype = cond_latents.dtype

        # Generate condition positions
        if modality_key == "video":
            cond_positions = self._get_video_positions(
                num_frames=cond.num_frames,
                height=cond.height,
                width=cond.width,
                batch_size=batch_size,
                fps=cond.fps,
                device=device,
            )
        else:
            cond_positions = self._get_audio_positions(
                num_time_steps=cond_seq_len,
                batch_size=batch_size,
                device=device,
            )

        # Translate / rescale ref positions into the target's frame (video only).
        if modality_key == "video":
            spatial_sf = self.reference_spatial_scale_factor or 1
            temporal_sf = self.reference_temporal_scale_factor or 1
            if spatial_sf != 1 or temporal_sf != 1:
                cond_positions = cond_positions.clone()
                if temporal_sf != 1:
                    # Ref positions are already at the ref's effective fps (source_fps / S,
                    # stored by process_videos.py). Shift by (S - 1) / target_fps so ref's
                    # last patch aligns with target's last; clamp the causal patch at 0.
                    t_target = positions[:, 0, 0:1, 1:2]  # = 1 / target_fps
                    cond_positions[:, 0, ...] = torch.clamp(
                        cond_positions[:, 0, ...] - (temporal_sf - 1) * t_target, min=0
                    )
                if spatial_sf != 1:
                    cond_positions[:, 1, ...] *= spatial_sf
                    cond_positions[:, 2, ...] *= spatial_sf

        # Condition tokens: clean, timestep=0, no loss
        cond_timesteps = torch.zeros(batch_size, cond_seq_len, device=device, dtype=dtype)
        cond_loss_mask = torch.zeros(batch_size, cond_seq_len, dtype=torch.bool, device=device)

        # Concatenate condition and target sequences (condition first, then target)
        combined_latents = torch.cat([cond_latents, noisy_latents], dim=1)
        combined_positions = torch.cat([cond_positions, positions], dim=2)
        combined_timesteps = torch.cat([cond_timesteps, timesteps], dim=1)

        combined_loss_mask = torch.cat([cond_loss_mask, loss_mask], dim=1) if loss_mask is not None else None

        # Targets remain unchanged (only for target portion, not condition portion)

        return combined_latents, combined_positions, combined_timesteps, combined_loss_mask, targets

    @staticmethod
    def _compute_modality_loss(pred: Tensor, targets: Tensor, loss_mask: Tensor) -> Tensor:
        """Compute per-element MSE loss for a single modality. Returns [B,]."""
        # Slice prediction to match targets length (removes any prepended condition tokens)
        target_len = targets.shape[1]
        pred = pred[:, -target_len:, :]
        mask = loss_mask[:, -target_len:]

        # Compute masked MSE loss, reduce per-element [B,] over (seq, channels)
        mask_expanded = mask.unsqueeze(-1).float()
        squared_error = (pred - targets).pow(2)
        masked_loss = squared_error * mask_expanded
        return masked_loss.mean(dim=[-2, -1]) / mask_expanded.mean(dim=[-2, -1]).clamp(min=1e-8)

    @staticmethod
    def _infer_scale_factor(cond_height: int, cond_width: int, target_height: int, target_width: int) -> int:
        """Infer spatial scale factor between condition and target resolutions."""
        if target_height == cond_height and target_width == cond_width:
            return 1
        scale_h = target_height // cond_height if cond_height > 0 else 1
        scale_w = target_width // cond_width if cond_width > 0 else 1
        if scale_h != scale_w:
            raise ValueError(
                f"Non-uniform scale factors between condition and target: height={scale_h}, width={scale_w}. "
                "Condition and target resolutions must scale uniformly."
            )
        return scale_h

    @staticmethod
    def _infer_temporal_scale_factor(cond_num_frames: int, target_num_frames: int) -> int:
        """Infer temporal scale factor between condition and target latent frame counts.
        The first latent frame encodes a single pixel frame (the VAE's causal structure),
        so the temporal groups count is (num_frames - 1).  The scale factor is the ratio
        of target groups to condition groups.
        """
        if target_num_frames == cond_num_frames:
            return 1
        target_groups = target_num_frames - 1
        cond_groups = cond_num_frames - 1
        if cond_groups <= 0 or target_groups <= 0:
            return 1
        if target_groups % cond_groups != 0:
            raise ValueError(
                f"Target temporal groups ({target_groups}) is not evenly divisible by "
                f"condition temporal groups ({cond_groups})."
            )
        return target_groups // cond_groups
