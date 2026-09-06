#!/usr/bin/env python3

"""Encode robot-action arrays into latents for WAM training.

The third leg of the precompute, alongside ``process_dataset.py``, which handles video latents
and text conditions. This one reads the ``actions`` column of the same manifest, runs each
clip's ``(num_actions, action_dim)`` array through the frozen action VAE, and writes the
whitened latents as ``.pt``.

Output paths mirror ``process_videos.py`` exactly -- the media path relative to the manifest's
directory, with a ``.pt`` suffix -- because ``PrecomputedDataset`` pairs its sources by relative
path and rejects mismatched file counts. So a clip whose video latent lands at::

    .precomputed/latents/videos_15fps/episode_000010_w00000.pt

gets its actions at::

    .precomputed/action_latents/videos_15fps/episode_000010_w00000.pt

Both normalization steps happen here and nowhere else: ``ActionProcessor`` scales raw
millimetres, radians and gripper counts into [-1, 1], and ``PerChannelStatistics`` whitens the
latent. Training loads the result as-is; the decoder is only needed at deploy.

Example:
    python scripts/process_actions.py \\
        /mnt/.../TD-WAM-720p-8tasks/dataset_win25.json \\
        --action-vae runs/avae_l7_b3e-5_h64
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn, TimeRemainingColumn

from ltx_core.model.action_vae.loading import load_action_encoder

app = typer.Typer(pretty_exceptions_enable=False)
console = Console()


def _output_relative(path: Path, data_root: Path) -> Path:
    """Relative path used to name a sample's cached output, mirroring the input layout.

    Copied from ``process_videos.py`` rather than imported, since that module pulls in torchaudio,
    pillow_heif and the whole video stack for a four-line helper. The two must stay in step: if
    the naming there changes, the action latents stop pairing with the video ones and
    ``PrecomputedDataset._validate_setup`` will complain about mismatched counts.
    """
    try:
        return path.relative_to(data_root)
    except ValueError:
        return Path(*path.parts[1:]) if path.is_absolute() else path


def _load_manifest(dataset_file: Path) -> pd.DataFrame:
    """Read the manifest in whichever of the supported formats it uses."""
    suffix = dataset_file.suffix.lower()
    if suffix == ".json":
        return pd.DataFrame(json.loads(dataset_file.read_text()))
    if suffix == ".jsonl":
        return pd.read_json(dataset_file, lines=True)
    if suffix == ".csv":
        return pd.read_csv(dataset_file)
    raise typer.BadParameter(f"unsupported manifest format '{suffix}'; expected .json, .jsonl or .csv")


@app.command()
def main(
    dataset_path: str = typer.Argument(
        ...,
        help="The same manifest passed to process_dataset.py. Needs an 'actions' column holding "
        "one .npy path per clip, and a 'video' column, which is what names the outputs.",
    ),
    action_vae: str = typer.Option(
        ...,
        help="Directory holding action_vae.safetensors and config.json, as written by "
        "train_action_vae.py.",
    ),
    output_dir: str | None = typer.Option(
        default=None,
        help="Output directory. Defaults to .precomputed beside the manifest, matching "
        "process_dataset.py, so the action latents sit next to the video ones.",
    ),
    action_column: str = typer.Option(default="actions", help="Manifest column holding the .npy paths."),
    video_column: str = typer.Option(default="video", help="Manifest column the output names are derived from."),
    subdir: str = typer.Option(default="action_latents", help="Subdirectory under the output directory."),
    device: str = typer.Option(default="cpu", help="Where to run the encoder. It is a two-layer MLP; cpu is fine."),
    overwrite: bool = typer.Option(default=False, help="Re-encode clips whose output already exists."),
) -> None:
    """Encode each clip's action array into whitened latents for WAM training."""
    dataset_file = Path(dataset_path).resolve()
    if not dataset_file.is_file():
        raise typer.BadParameter(f"no manifest at {dataset_file}")
    data_root = dataset_file.parent

    frame = _load_manifest(dataset_file)
    for column in (action_column, video_column):
        if column not in frame.columns:
            raise typer.BadParameter(f"manifest has no '{column}' column; found {list(frame.columns)}")
    missing_actions = int(frame[action_column].isna().sum())
    if missing_actions:
        raise typer.BadParameter(
            f"{missing_actions} of {len(frame)} manifest rows have no action array. Re-run "
            "build_ltx_manifest.py with --actions so every clip has one."
        )

    output_base = Path(output_dir) if output_dir else data_root / ".precomputed"
    output_path = output_base / subdir
    output_path.mkdir(parents=True, exist_ok=True)

    with console.status(f"[bold]Loading action VAE from [cyan]{action_vae}[/]...", spinner="dots"):
        encoder = load_action_encoder(action_vae, device=device)
    console.print(
        f"Action VAE: action_dim={encoder.action_dim}, latent_channels={encoder.latent_channels}, device={device}"
    )

    written = skipped = 0
    expected_shape: tuple[int, int] | None = None

    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Encoding actions", total=len(frame))
        for row in frame.itertuples(index=False):
            progress.advance(task)

            video_file = Path(str(getattr(row, video_column)))
            destination = output_path / _output_relative(video_file, data_root).with_suffix(".pt")
            if destination.is_file() and not overwrite:
                skipped += 1
                continue

            actions = np.load(Path(str(getattr(row, action_column))))
            if actions.ndim != 2:
                raise ValueError(f"expected a (num_actions, action_dim) array, got shape {actions.shape}")
            if actions.shape[1] != encoder.action_dim:
                raise ValueError(
                    f"action array has {actions.shape[1]} dimensions but the VAE expects "
                    f"{encoder.action_dim}: {video_file}"
                )
            # Every clip must yield the same number of action tokens -- the sequence layout in
            # _interleave_action_tokens assumes it, and a ragged batch would fail to collate.
            if expected_shape is None:
                expected_shape = actions.shape
            elif actions.shape != expected_shape:
                raise ValueError(
                    f"clip has {actions.shape[0]} actions but earlier clips have "
                    f"{expected_shape[0]}: {video_file}"
                )

            with torch.no_grad():
                batch = torch.from_numpy(actions).to(device=device, dtype=torch.float32).unsqueeze(0)
                latents = encoder(batch).squeeze(0)

            destination.parent.mkdir(parents=True, exist_ok=True)
            torch.save(latents.cpu(), destination)
            written += 1

    console.print(f"[green]Wrote {written} action latents to {output_path}[/] ({skipped} already present)")
    if expected_shape is not None:
        console.print(f"Each is ({expected_shape[0]}, {encoder.latent_channels}) from a {expected_shape} array")


if __name__ == "__main__":
    app()
