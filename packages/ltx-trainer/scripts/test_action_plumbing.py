#!/usr/bin/env python3

"""Check that action tokens actually route through action_in and action_out.

Everything tested so far has been the *layout* -- which indices are actions, that interleave and
extract are inverses, that the refactor matched the old code. None of it touched the model. The
entry and exit paths (``apply_action_projection`` on the way in, ``action_out`` and
``action_scale_shift_table`` on the way out) were written, reviewed by eye, and never run.

That gap matters right now: ``action_out`` has barely moved from its initialization after 4,000
steps, which is equally consistent with "wired correctly, under-trained" and with "the output at
those positions barely depends on action_out at all".

Builds a small LTXModel on the CPU -- 2 layers, 4 heads, 64-dim -- so this runs in seconds beside
a training job without touching a GPU. The checks are about wiring, and wiring does not care how
big the model is.

    python test_action_plumbing.py

Every check prints PASS or FAIL and the number it is based on.
"""

from __future__ import annotations

import torch

from ltx_core.action_layout import ActionTokenLayout
from ltx_core.model.transformer.model import LTXModel
from ltx_core.model.transformer.modality import Modality
from ltx_core.types import LTXModelType

CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, passed: bool, detail: str) -> None:
    CHECKS.append((name, passed, detail))
    print(f"  {'PASS' if passed else 'FAIL'}  {name}\n        {detail}")


def build() -> tuple[LTXModel, ActionTokenLayout]:
    """A small video-only model with an action stream, and the layout for its sequence."""
    torch.manual_seed(0)
    model = LTXModel(
        model_type=LTXModelType.Video,
        num_attention_heads=4,
        attention_head_dim=16,
        in_channels=128,
        out_channels=128,
        num_layers=2,
        cross_attention_dim=64,
        action_channels=7,
    ).eval()
    layout = ActionTokenLayout(
        num_actions=8, num_frames=3, tokens_per_frame=6, temporal_scale=4, fps=15.0
    )
    return model, layout


def make_modality(layout: ActionTokenLayout, model: LTXModel, batch: int = 1) -> Modality:
    """A sequence shaped exactly as the trainer builds it, with the action mask set."""
    device = torch.device("cpu")
    total = layout.total_tokens
    torch.manual_seed(1)
    positions = torch.rand(batch, 3, total, 2)
    positions[:, :, layout.action_indices] = layout.positions(batch, device)[:, :, :, :]
    return Modality(
        enabled=True,
        latent=torch.randn(batch, total, 128),
        sigma=torch.full((batch,), 0.5),
        timesteps=torch.full((batch, total), 0.5),
        positions=positions,
        context=torch.randn(batch, 12, 64),
        context_mask=None,
        action_mask=layout.mask(batch, device),
    )


@torch.no_grad()
def run_forward(model: LTXModel, modality: Modality) -> torch.Tensor:
    video_out, _ = model(video=modality, audio=None, perturbations=None)
    return video_out


def main() -> None:
    model, layout = build()
    modality = make_modality(layout, model)
    action_idx = list(layout.action_indices)
    video_idx = list(layout.video_indices)

    print(f"\nsequence: {layout.total_tokens} tokens = {layout.video_tokens} video + {layout.num_actions} action")
    print(f"action positions: {action_idx}\n")

    base = run_forward(model, modality)
    print(f"output shape: {tuple(base.shape)}\n")

    # 1. The entry projection. Perturbing action_in must change the action positions and leave
    #    the video ones alone -- attention mixes tokens, so a change at an action position can
    #    leak into video positions; what must hold is that the action positions move *more*.
    with torch.no_grad():
        saved = model.action_in.weight.clone()
        model.action_in.weight.add_(torch.randn_like(saved))
    perturbed = run_forward(model, modality)
    with torch.no_grad():
        model.action_in.weight.copy_(saved)

    delta = (perturbed - base).abs()
    on_action = delta[:, action_idx].mean().item()
    on_video = delta[:, video_idx].mean().item()
    check(
        "action_in feeds the action positions",
        on_action > 10 * max(on_video, 1e-12),
        f"perturbing action_in moved action positions by {on_action:.4e}, video by {on_video:.4e}",
    )

    # 2. The exit projection. action_out is applied only at action positions, and nothing else
    #    writes there, so perturbing it must move those and leave video *exactly* untouched.
    with torch.no_grad():
        saved = model.action_out.weight.clone()
        model.action_out.weight.add_(torch.randn_like(saved))
    perturbed = run_forward(model, modality)
    with torch.no_grad():
        model.action_out.weight.copy_(saved)

    delta = (perturbed - base).abs()
    on_action = delta[:, action_idx].mean().item()
    on_video = delta[:, video_idx].max().item()
    check(
        "action_out writes the action positions",
        on_action > 0,
        f"perturbing action_out moved action positions by {on_action:.4e}",
    )
    check(
        "action_out leaves video untouched",
        on_video == 0.0,
        f"largest change at any video position: {on_video:.4e} (want exactly 0)",
    )

    # 3. Only the first action_channels of an action row carry the prediction; the rest is the
    #    zero padding the row was widened with, and the loss slices it off.
    with torch.no_grad():
        saved = model.action_out.weight.clone()
        model.action_out.weight.add_(torch.randn_like(saved))
    perturbed = run_forward(model, modality)
    with torch.no_grad():
        model.action_out.weight.copy_(saved)

    tail = (perturbed - base)[:, action_idx, 7:].abs().max().item()
    check(
        "the padded channels stay padding",
        tail == 0.0,
        f"largest change beyond channel 7 of an action row: {tail:.4e} (want exactly 0)",
    )

    # 4. The scale-shift table. Zero-initialized, so it contributes nothing until trained --
    #    but it must be *able* to, or it would be dead weight the optimizer updates for nothing.
    with torch.no_grad():
        saved = model.action_scale_shift_table.clone()
        model.action_scale_shift_table.add_(torch.randn_like(saved))
    perturbed = run_forward(model, modality)
    with torch.no_grad():
        model.action_scale_shift_table.copy_(saved)

    delta = (perturbed - base).abs()
    check(
        "action_scale_shift_table reaches the output",
        delta[:, action_idx].mean().item() > 0,
        f"perturbing the table moved action positions by {delta[:, action_idx].mean().item():.4e}",
    )

    # 5. Gradients. The parameters can influence the output; this asks whether autograd carries
    #    a signal back to them from a loss on the action positions alone -- which is what the
    #    training loss actually is.
    model.zero_grad(set_to_none=True)
    out, _ = model(video=make_modality(layout, model), audio=None, perturbations=None)
    out[:, action_idx, :7].pow(2).mean().backward()

    for name, param in [
        ("action_in.weight", model.action_in.weight),
        ("action_out.weight", model.action_out.weight),
        ("action_scale_shift_table", model.action_scale_shift_table),
    ]:
        grad = param.grad
        norm = float(grad.norm()) if grad is not None else 0.0
        check(f"{name} receives gradient", grad is not None and norm > 0, f"grad norm {norm:.4e}")

    # 6. And the converse: a loss on the video positions alone must not train the action head,
    #    or the two streams are entangled in a way the loss split assumes they are not.
    model.zero_grad(set_to_none=True)
    out, _ = model(video=make_modality(layout, model), audio=None, perturbations=None)
    out[:, video_idx].pow(2).mean().backward()

    grad = model.action_out.weight.grad
    norm = float(grad.norm()) if grad is not None else 0.0
    check(
        "a video-only loss does not reach action_out",
        norm == 0.0,
        f"action_out grad norm from a video-only loss: {norm:.4e} (want exactly 0)",
    )

    failed = [name for name, passed, _ in CHECKS if not passed]
    print()
    if failed:
        print(f"FAILED: {', '.join(failed)}")
        raise SystemExit(1)
    print(f"all {len(CHECKS)} checks passed")


if __name__ == "__main__":
    main()
