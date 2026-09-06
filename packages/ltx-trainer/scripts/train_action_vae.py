#!/usr/bin/env python3
"""Train the per-step action VAE for the World Action Model.

Reads raw actions out of a LeRobot dataset's parquet files, fits the per-dimension scale on
the training split, trains a small KL VAE, then fits the dataset-level whitening buffers the
same way LTX does for video and audio (statistics of the posterior means, stored inside the
VAE state_dict). The result is frozen and consumed by the WAM finetune.

Scaling: the six pose deltas are scaled by the training split's min/max and never clamped,
so a fast motion in held-out data lands slightly outside [-1, 1] instead of being flattened.
The gripper is scaled by its physical travel (--gripper-range, default 0-850) and clamped,
the same way the video VAE divides pixels by 255 rather than by a dataset statistic.

Reduction convention -- this is what makes ``--kl-weight`` mean something:
    reconstruction = squared error SUMMED over the action dimensions, averaged over samples
    kl            = KL SUMMED over the latent dimensions, averaged over samples
so ``loss = recon + beta * kl`` with ``beta = 2 * sigma_obs^2``. Do not trust that as a way to
pick beta -- sweep it against the reported error in raw units.

Defaults are the settled configuration, chosen from a 24-run sweep over latent width, beta and
hidden width:

* **latent 7.** Exactly 7 dimensions stayed active at every beta below 1e-2 and at every latent
  width tried (7, 8, 16) -- dead dimensions read exactly 0.000 nats, so the count is not an
  artifact of a threshold. PCA on the normalized actions confirms full rank 7 (smallest
  correlation eigenvalue 0.30). Wider latents carry inert channels whose whitening std is
  degenerate.
* **beta 3e-5.** Held-out error falls 0.209 -> 0.064 -> 0.039 mm across 3e-4 / 1e-4 / 3e-5, then
  rises again at 1e-5. 3e-5 is the floor. This is far from a KL-dominated regime by the
  standards of the field -- LDM's AutoencoderKL uses 1e-6 -- and information the VAE discards
  cannot be recovered by anything downstream, whereas distribution mismatch can be, since
  ``action_in`` and the LoRA adapters are trained.
* **hidden 64, 2 layers.** No measured benefit from 128. Three seeds at the chosen config gave
  0.064 +/- 0.011 mm, a ~17% spread that swamps every width difference observed.

Example:
    python scripts/train_action_vae.py \
        --dataset /mnt/filestore/hf-cache/lerobot/bellboy-robotics/TD-WAM-720p-8tasks \
        --held-out-episodes held_out.txt \
        --out runs/action_vae_l7_b3e-5
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from safetensors.torch import save_file

from ltx_core.model.action_vae import (
    ActionDecoder,
    ActionEncoder,
    kl_divergence,
    reparameterize,
)

logger = logging.getLogger("train_action_vae")

DIM_NAMES = ["dx", "dy", "dz", "rx", "ry", "rz", "gripper"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", type=Path, required=True, help="LeRobot dataset root (contains data/).")
    p.add_argument("--action-column", default="action", help="Parquet column holding the action vector.")
    p.add_argument(
        "--held-out-episodes",
        type=Path,
        default=None,
        help="File with one held-out episode_index per line. These episodes are excluded from "
        "training and from the range / whitening fits, and used for evaluation.",
    )
    p.add_argument("--held-out-frac", type=float, default=0.02, help="Used only if --held-out-episodes is absent.")

    p.add_argument("--latent-channels", type=int, default=7)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--kl-weight", type=float, default=3e-5)
    p.add_argument(
        "--gripper-range",
        type=float,
        nargs=2,
        metavar=("LOW", "HIGH"),
        default=[0.0, 850.0],
        help="Physical travel of the gripper, used instead of the fitted min/max and clamped. "
        "The six pose deltas have no physical bound and are always fitted from the data.",
    )
    p.add_argument("--gripper-index", type=int, default=-1, help="Which action dimension is the gripper.")
    p.add_argument(
        "--fit-gripper",
        action="store_true",
        help="Fit the gripper from the data like the deltas instead of using --gripper-range.",
    )

    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", type=Path, required=True, help="Output directory.")
    return p.parse_args()


def load_actions(dataset: Path, action_column: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (actions (N, D), episode_index (N,)) from every parquet under ``dataset/data``."""
    files = sorted((dataset / "data").rglob("*.parquet"))
    if not files:
        sys.exit(f"no parquet files under {dataset / 'data'}")
    logger.info("reading %d parquet files", len(files))

    actions, episodes = [], []
    for i, path in enumerate(files, start=1):
        df = pd.read_parquet(path, columns=[action_column, "episode_index"])
        actions.append(np.stack(df[action_column].to_numpy()).astype(np.float32))
        episodes.append(df["episode_index"].to_numpy())
        if i % 200 == 0:
            logger.info("  %d/%d", i, len(files))

    return np.concatenate(actions), np.concatenate(episodes)


def split_episodes(episodes: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    """Boolean mask, True where the row belongs to a held-out episode."""
    unique = np.unique(episodes)
    if args.held_out_episodes is not None:
        text = args.held_out_episodes.read_text().split()
        held = {int(tok.strip(",")) for tok in text if tok.strip(",")}
        missing = held - set(unique.tolist())
        if missing:
            sys.exit(f"held-out episodes not present in the dataset: {sorted(missing)}")
    else:
        rng = np.random.default_rng(args.seed)
        count = max(1, int(round(len(unique) * args.held_out_frac)))
        held = set(rng.choice(unique, size=count, replace=False).tolist())
        logger.warning("no --held-out-episodes given; randomly holding out %d episodes", count)
    logger.info("held-out episodes: %d of %d", len(held), len(unique))
    return np.isin(episodes, list(held))


def report(
    encoder: ActionEncoder,
    decoder: ActionDecoder,
    actions: torch.Tensor,
    label: str,
    dim_names: list[str],
) -> dict:
    """Reconstruction error in raw units and per-dimension KL in nats.

    The error is measured against the *representable* target, not the raw action. A clamped
    dimension (the gripper) cannot represent readings outside its physical range, so the
    round-trip ``un_normalize(normalize(x))`` is the best any model could do. Charging that
    gap to the VAE would hide its real error behind the clamp, so it is reported separately.
    """
    encoder.eval()
    decoder.eval()
    processor = encoder.action_processor
    with torch.no_grad():
        batch = actions.unsqueeze(1)  # (N, 1, D)
        target = processor.un_normalize(processor.normalize(batch))
        means, logvar = encoder.encode_moments(batch)
        # Evaluate the deterministic path -- that is what the finetune will use.
        recon = decoder(encoder.per_channel_statistics.normalize(means))
        error = (recon - target).squeeze(1)
        clamp_loss = (target - batch).abs().squeeze(1)
        kl_per_dim = kl_divergence(means, logvar).squeeze(1).mean(dim=0)

    mae = error.abs().mean(dim=0)
    rmse = error.pow(2).mean(dim=0).sqrt()
    clamp_mae = clamp_loss.mean(dim=0)
    clamp_max = clamp_loss.max(dim=0).values

    logger.info("--- %s (n=%d) ---", label, actions.shape[0])
    logger.info("%-8s %14s %14s %14s %14s", "dim", "MAE (raw)", "RMSE (raw)", "clamp MAE", "clamp max")
    for i, name in enumerate(dim_names):
        logger.info(
            "%-8s %14.6g %14.6g %14.6g %14.6g",
            name,
            mae[i].item(),
            rmse[i].item(),
            clamp_mae[i].item(),
            clamp_max[i].item(),
        )
    logger.info(
        "per-dim KL (nats): %s",
        " ".join(f"{v:.3f}" for v in kl_per_dim.tolist()),
    )
    dead = [i for i, v in enumerate(kl_per_dim.tolist()) if v < 0.1]
    if dead:
        logger.warning(
            "latent dims %s carry < 0.1 nats -- they are collapsed. Shrink --latent-channels "
            "or lower --kl-weight; collapsed dims make the whitening std degenerate.",
            dead,
        )
    return {
        "n": int(actions.shape[0]),
        "mae": mae.tolist(),
        "rmse": rmse.tolist(),
        "clamp_mae": clamp_mae.tolist(),
        "clamp_max": clamp_max.tolist(),
        "kl_per_dim": kl_per_dim.tolist(),
        "dead_dims": dead,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    torch.manual_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    actions_np, episodes_np = load_actions(args.dataset, args.action_column)
    action_dim = actions_np.shape[1]
    dim_names = DIM_NAMES[:action_dim] if action_dim <= len(DIM_NAMES) else [f"d{i}" for i in range(action_dim)]
    logger.info("loaded %d actions, action_dim=%d", len(actions_np), action_dim)

    held_mask = split_episodes(episodes_np, args)
    train = torch.from_numpy(actions_np[~held_mask]).to(args.device)
    evaluation = torch.from_numpy(actions_np[held_mask]).to(args.device)
    logger.info("train rows: %d, held-out rows: %d", len(train), len(evaluation))

    encoder = ActionEncoder(action_dim, args.latent_channels, args.hidden, args.num_layers).to(args.device)
    decoder = ActionDecoder(action_dim, args.latent_channels, args.hidden, args.num_layers).to(args.device)

    # --- Fit the raw-action range on the training split only. ---
    gripper_range = None if args.fit_gripper else tuple(args.gripper_range)
    stats = encoder.action_processor.fit(train, gripper_range=gripper_range, gripper_index=args.gripper_index)
    decoder.action_processor.load_state_dict(encoder.action_processor.state_dict())

    lo = encoder.action_processor.get_buffer("range-low").tolist()
    hi = encoder.action_processor.get_buffer("range-high").tolist()
    clamped = encoder.action_processor.get_buffer("clamp-mask").tolist()

    # The units are readable straight off this table: translation around +/-20 is mm,
    # around +/-0.02 would be metres. The ratio column is the check on min/max being a safe
    # scale -- a few x means the tail is real motion, 20x or more means sensor spikes and
    # the fit should move to quantiles instead.
    logger.info("raw action range fitted on the training split:")
    logger.info("%-8s %13s %13s %13s %13s %9s %8s", "dim", "lo", "hi", "data min", "data max", "|p99|", "max/p99")
    for i, name in enumerate(dim_names):
        ratio = stats["max"][i] / max(stats["p99"][i], 1e-12)
        logger.info(
            "%-8s %13.6g %13.6g %13.6g %13.6g %9.4g %7.1fx%s",
            name,
            lo[i],
            hi[i],
            stats["min"][i],
            stats["max"][i],
            stats["p99"][i],
            ratio,
            "  (clamped, physical bound)" if clamped[i] > 0.5 else "",
        )

    # --- Train. ---
    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr)
    generator = torch.Generator(device=args.device).manual_seed(args.seed)

    encoder.train()
    decoder.train()
    for step in range(1, args.steps + 1):
        index = torch.randint(0, len(train), (args.batch_size,), device=args.device, generator=generator)
        batch = train[index].unsqueeze(1)  # (B, 1, D)
        target = encoder.action_processor.normalize(batch)

        means, logvar = encoder.encode_moments(batch)
        latent = reparameterize(means, logvar)
        recon = decoder.decode_normalized(latent)

        # Summed over the feature axis, averaged over samples -- see the module docstring.
        recon_loss = (recon - target).pow(2).sum(dim=-1).mean()
        kl_loss = kl_divergence(means, logvar).sum(dim=-1).mean()
        loss = recon_loss + args.kl_weight * kl_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % 1000 == 0 or step == 1:
            logger.info(
                "step %6d  loss %.6f  recon %.6f  kl %.4f nats",
                step,
                loss.item(),
                recon_loss.item(),
                kl_loss.item(),
            )

    # --- Fit the dataset-level whitening on the posterior means of the training split. ---
    encoder.eval()
    with torch.no_grad():
        chunks = [encoder.encode_moments(train[i : i + 65536].unsqueeze(1))[0].squeeze(1) for i in range(0, len(train), 65536)]
        train_means = torch.cat(chunks)
    encoder.per_channel_statistics.fit(train_means)
    decoder.per_channel_statistics.load_state_dict(encoder.per_channel_statistics.state_dict())
    logger.info("mean-of-means: %s", encoder.per_channel_statistics.get_buffer("mean-of-means").tolist())
    logger.info("std-of-means:  %s", encoder.per_channel_statistics.get_buffer("std-of-means").tolist())

    metrics = {
        "train": report(encoder, decoder, train, "train", dim_names),
        "held_out": report(encoder, decoder, evaluation, "held-out", dim_names),
    }

    # --- Save in the LTX checkpoint layout, so an SDOps prefix filter can load it. ---
    state: dict[str, torch.Tensor] = {}
    for key, value in encoder.net.state_dict().items():
        state[f"action_vae.encoder.{key}"] = value.cpu()
    for key, value in decoder.net.state_dict().items():
        state[f"action_vae.decoder.{key}"] = value.cpu()
    for key, value in encoder.per_channel_statistics.state_dict().items():
        state[f"action_vae.per_channel_statistics.{key}"] = value.cpu()
    for key, value in encoder.action_processor.state_dict().items():
        state[f"action_vae.action_processor.{key}"] = value.cpu()
    save_file(state, str(args.out / "action_vae.safetensors"))

    config = {
        "action_dim": action_dim,
        "latent_channels": args.latent_channels,
        "hidden": args.hidden,
        "num_layers": args.num_layers,
        "kl_weight": args.kl_weight,
        "gripper_range": None if args.fit_gripper else list(args.gripper_range),
        "gripper_index": args.gripper_index,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "seed": args.seed,
        "dim_names": dim_names,
        "range_low": lo,
        "range_high": hi,
        "clamp_mask": clamped,
        "data_stats": stats,
        "held_out_episodes": sorted(set(episodes_np[held_mask].tolist())),
        "metrics": metrics,
    }
    (args.out / "config.json").write_text(json.dumps(config, indent=2))
    logger.info("wrote %s", args.out / "action_vae.safetensors")


if __name__ == "__main__":
    main()
