#!/usr/bin/env python3

"""Deduplicate the caption embeddings by hardlinking clips from the same episode.

The caption pipeline writes one embedding file per clip. That suits a text-to-video dataset
where every clip has its own caption; ours has one task instruction per episode, repeated
across that episode's ~40 windows. At 13 MB per file, the full training set would need about
378 GB to store roughly 800 copies of each tensor.

A hardlink is a second directory entry for the same bytes, so the clips of one episode can
share a single file. The loader opens a path and reads a tensor; it neither knows nor cares
that the inode is shared. Cost drops to one file per episode -- about 10.7 GB.

Grouping is by episode index parsed from the clip filename, not by comparing caption strings.
The episode is structural: ``build_ltx_manifest.py`` assigns one caption per episode, so clips
of an episode share a caption by construction. Two episodes that happen to word their captions
identically simply get two copies, which costs 13 MB and cannot go wrong.

Run in three steps::

    # 1. one representative clip per episode
    python3 dedup_captions.py reps  DATASET.json --out reps.json

    # 2. encode only those (821 instead of 32,252)
    uv run python packages/ltx-trainer/scripts/process_captions.py reps.json \\
        --text-encoder-path .../ltx-2.5-22b-gemma4-12b \\
        --output-dir DATASET_DIR/.precomputed/conditions

    # 3. link every other clip to its episode's file
    python3 dedup_captions.py link DATASET.json

Afterwards every clip's path exists, so ``process_dataset.py`` skips the caption phase
entirely (its check is ``output.is_file()``) and goes straight to the video latents.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

EPISODE_PATTERN = re.compile(r"episode_(\d+)")


def load_manifest(path: Path) -> list[dict]:
    """Read the manifest, which build_ltx_manifest.py writes as a JSON list."""
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return json.loads(path.read_text())


def output_relative(video: Path, data_root: Path) -> Path:
    """Where the precompute names this clip's output.

    Mirrors ``_output_relative`` in ``process_videos.py``: the media path relative to the
    manifest's directory. Must stay in step with it, or the links land on names nothing reads.
    """
    try:
        return video.relative_to(data_root)
    except ValueError:
        return Path(*video.parts[1:]) if video.is_absolute() else video


def group_by_episode(records: list[dict]) -> dict[str, list[dict]]:
    """Group manifest rows by the episode index in their clip filename."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        video = Path(record["video"])
        match = EPISODE_PATTERN.search(video.name)
        if not match:
            sys.exit(f"cannot read an episode index from '{video.name}'; expected episode_<digits>_...")
        groups[match.group(1)].append(record)
    return groups


def check_one_caption_per_episode(groups: dict[str, list[dict]]) -> None:
    """Fail if an episode's clips disagree about the caption.

    The whole scheme rests on this. If it ever stops holding, linking would silently attach one
    clip's text to forty others -- a mistake that would not surface until the model was trained
    on it.
    """
    for episode, records in sorted(groups.items()):
        captions = {record["caption"] for record in records}
        if len(captions) > 1:
            sys.exit(
                f"episode {episode} has {len(captions)} different captions across its "
                f"{len(records)} clips, so they cannot share one embedding:\n  "
                + "\n  ".join(sorted(captions)[:5])
            )


def command_reps(args: argparse.Namespace) -> None:
    """Write a manifest holding one clip per episode."""
    manifest = Path(args.manifest).resolve()
    records = load_manifest(manifest)
    groups = group_by_episode(records)
    check_one_caption_per_episode(groups)

    reps = [records_for[0] for _, records_for in sorted(groups.items())]
    out = Path(args.out).resolve() if args.out else manifest.with_name(manifest.stem + "_reps.json")
    if out.parent != manifest.parent:
        sys.exit(
            f"the representative manifest must sit beside the original ({manifest.parent}), "
            "because output names are derived from the manifest's directory"
        )
    out.write_text(json.dumps(reps, indent=2))

    print(f"{len(records)} clips over {len(groups)} episodes")
    print(f"distinct captions: {len({r['caption'] for r in records})}")
    print(f"wrote {len(reps)} representatives -> {out}")


def command_link(args: argparse.Namespace) -> None:
    """Hardlink every clip to its episode's already-encoded embedding."""
    manifest = Path(args.manifest).resolve()
    data_root = manifest.parent
    conditions = Path(args.conditions).resolve() if args.conditions else data_root / ".precomputed" / "conditions"
    if not conditions.is_dir():
        sys.exit(f"no conditions directory at {conditions}; run the caption step first")

    records = load_manifest(manifest)
    groups = group_by_episode(records)
    check_one_caption_per_episode(groups)

    linked = existing = 0
    missing: list[str] = []

    for episode, episode_records in sorted(groups.items()):
        source = conditions / output_relative(Path(episode_records[0]["video"]), data_root).with_suffix(".pt")
        if not source.is_file():
            missing.append(f"episode {episode}: {source}")
            continue

        for record in episode_records[1:]:
            destination = conditions / output_relative(Path(record["video"]), data_root).with_suffix(".pt")
            if destination.exists():
                existing += 1
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            # Hardlink, not copy: one inode, many names. Same filesystem is required, which
            # holds since both sit under the same conditions directory.
            os.link(source, destination)
            linked += 1

    if missing:
        print(f"{len(missing)} episodes have no encoded representative:", file=sys.stderr)
        for line in missing[:10]:
            print(f"  {line}", file=sys.stderr)
        sys.exit("run the caption step on the representative manifest first")

    total = sum(len(v) for v in groups.values())
    print(f"linked {linked}, already present {existing}, {len(groups)} real files for {total} clips")
    print(f"storage: {len(groups)} files instead of {total} -- a factor of {total / len(groups):.0f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    reps = sub.add_parser("reps", help="write a manifest with one clip per episode")
    reps.add_argument("manifest")
    reps.add_argument("--out", default=None, help="defaults to <manifest>_reps.json beside the original")
    reps.set_defaults(func=command_reps)

    link = sub.add_parser("link", help="hardlink each clip to its episode's embedding")
    link.add_argument("manifest")
    link.add_argument("--conditions", default=None, help="defaults to <manifest dir>/.precomputed/conditions")
    link.set_defaults(func=command_link)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
