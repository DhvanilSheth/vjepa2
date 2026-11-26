# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Frozen-feature object-tracking probe for V-JEPA 3 checkpoints (DAVIS 2017).

WHAT IT MEASURES
----------------
Semi-supervised video object segmentation by label propagation: the objects
are outlined in the first frame, and the probe tracks them through the video
using only the similarity of frozen patch features between frames (see
propagate.py).  Nothing is trained, so there is nothing to tune per arm.  The
score is J&F-Mean: the average of J (mask overlap, IoU) and F (how well the
mask outline matches), over every object of every validation video.

This is the dense, temporal reading the clustering probe cannot give: it
rewards patch features that stay the same on one object and differ from the
background, frame after frame.

PROTOCOL
--------
V-JEPA 2.1's (paper Sections 7 and C.3): short side 480 px, patch 16, 15
context frames, circle neighbourhood of radius 12, top-5, temperature 0.2.
The paper reports 67.0 J&F for its distilled ViT-B and 69.0 for ViT-G.  A
proxy run of a few thousand iterations on a small dataset will be far below
that; the probe is for RANKING arms that were trained identically.

TWO WAYS OF FEEDING THE VIDEO (`feature_mode`)
----------------------------------------------
  frame  each frame alone, repeated to fill one tubelet.  Every frame gets
         its own feature map; the encoder sees no motion.  Closest to how
         image models are evaluated.
  clip   consecutive windows of `frames_per_clip` frames, as in training.
         The encoder sees motion and its temporal position encoding is
         exercised, but one feature map covers `tubelet_size` frames, so the
         two frames of a tubelet receive the same mask.
Positional arms that change only the TEMPORAL encoding are evaluated at a
single temporal position in `frame` mode; report `clip` mode for them too.

NOT COVERED
-----------
* The image patch embedding (T == 1) is never used: the single-GPU proxy
  recipe trains on video only, so that layer is untrained there.
* `clip` mode feeds consecutive DAVIS frames (24 fps); training sampled at
  4 fps.  The frame-rate mismatch is not corrected.
* `interpolate_rope` squeezes each spatial axis onto the 16-cell training
  grid, so on a non-square 480 x 848 input the two axes get different
  position spacing.  That is the model's own behaviour, left as is.

RUN (single GPU)
----------------
    python -m evals.video_tracking_frozen.eval \\
        --run /path/to/run_folder --davis /path/to/DAVIS

or through the common launcher with configs/eval_3/vitb16/davis-tracking.yaml.

STATUS
------
propagate.py, metrics.py, davis.py and the feature extraction are unit-tested
(tests/probes/test_tracking_probe.py), including an end-to-end run on a
synthetic DAVIS-layout folder.  It has not yet been run on the real DAVIS
set or a trained checkpoint, and the metrics are not yet cross-checked
against the official toolkit.
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from app.vjepa_3.pretrained import (
    RELEASED_2_1,
    read_encoder_state_dict,
    released_2_1_pretrain_config,
)
from evals.feature_clustering_frozen.eval import build_encoder, load_encoder_weights
from evals.video_tracking_frozen.davis import (
    load_frames,
    read_annotation,
    read_sequences,
    sequence_files,
)
from evals.video_tracking_frozen.metrics import boundary_f_measure, region_similarity
from evals.video_tracking_frozen.propagate import (
    downsample_labels,
    labels_to_mask,
    propagate_video,
)
from src.utils.logging import get_logger

logger = get_logger("video_tracking_frozen")

# Object size buckets, by the share of the first frame the object covers.
# Our own split (DAVIS defines none); it exists to show whether an arm helps
# small objects, which a 16-px patch grid resolves poorly.
SIZE_BUCKETS = (("small", 0.0, 0.01), ("medium", 0.01, 0.05), ("large", 0.05, 1.01))

DTYPES = {"float32": None, "float16": torch.float16, "bfloat16": torch.bfloat16}


@torch.no_grad()
def frame_features(
    encoder,
    frames,
    mode="frame",
    patch_size=16,
    tubelet_size=2,
    frames_per_clip=16,
    batch_size=8,
    autocast_dtype=None,
):
    """Frozen features for every frame: [N, 3, H, W] -> [N, h, w, D] float32.

    See the module docstring for the two modes.
    """
    n_frames, _, height, width = frames.shape
    h, w = height // patch_size, width // patch_size
    device = frames.device
    out = []

    def run(clip):
        with torch.autocast(
            device_type=device.type,
            dtype=autocast_dtype,
            enabled=autocast_dtype is not None,
        ):
            return encoder(clip).float()

    if mode == "frame":
        for chunk in frames.split(batch_size):
            clip = chunk[:, :, None].expand(-1, -1, tubelet_size, -1, -1)
            out.append(run(clip).reshape(len(chunk), h, w, -1))
    elif mode == "clip":
        if frames_per_clip % tubelet_size:
            raise ValueError("frames_per_clip must be a multiple of tubelet_size")
        for start in range(0, n_frames, frames_per_clip):
            window = frames[start : start + frames_per_clip]
            n_real = len(window)
            if n_real < frames_per_clip:  # last window: hold the final frame
                pad = window[-1:].expand(frames_per_clip - n_real, -1, -1, -1)
                window = torch.cat([window, pad])
            tokens = run(window.permute(1, 0, 2, 3)[None])  # [1, T/tub * h * w, D]
            maps = tokens.reshape(frames_per_clip // tubelet_size, h, w, -1)
            out.append(maps.repeat_interleave(tubelet_size, dim=0)[:n_real])
    else:
        raise ValueError(f"feature_mode must be 'frame' or 'clip', got {mode!r}")
    return torch.cat(out)


def first_frame_labels(annotation, size, patch_size):
    """Object ids of frame 0 -> per-patch label fractions [C, h, w].

    Channel 0 is background, channel i is object id i.  The annotation is
    resized to the model's input `size` first.
    """
    ids = torch.from_numpy(annotation.astype(np.int64))
    n_objects = int(ids.max())
    onehot = F.one_hot(ids, n_objects + 1).permute(2, 0, 1).float()
    onehot = F.interpolate(onehot[None], size=size, mode="bilinear", align_corners=False)[0]
    return downsample_labels(onehot, patch_size), n_objects


def scored_frames(n_frames):
    """Frame indices that count toward the score.  As in the official DAVIS
    evaluation, the first frame (its mask is given) and the last are left out."""
    return range(1, n_frames - 1)


@torch.no_grad()
def evaluate_sequence(encoder, davis_root, sequence, device, cfg):
    """Track one sequence; returns one record per object."""
    frame_paths, mask_paths = sequence_files(davis_root, sequence)
    frames, original_size = load_frames(frame_paths, cfg["short_side"], cfg["patch_size"])
    first = read_annotation(mask_paths[0])
    labels, n_objects = first_frame_labels(first, frames.shape[-2:], cfg["patch_size"])
    if n_objects == 0:
        return []

    feats = frame_features(
        encoder,
        frames.to(device),
        mode=cfg["feature_mode"],
        patch_size=cfg["patch_size"],
        tubelet_size=cfg["tubelet_size"],
        frames_per_clip=cfg["frames_per_clip"],
        batch_size=cfg["batch_size"],
        autocast_dtype=DTYPES[cfg["dtype"]],
    )
    soft = propagate_video(
        feats,
        labels.to(device),
        n_context=cfg["n_context"],
        size=cfg["neighborhood_size"],
        shape=cfg["neighborhood_shape"],
        topk=cfg["topk"],
        temperature=cfg["temperature"],
    )

    scored = scored_frames(len(frame_paths))
    j = np.zeros((n_objects, len(scored)))
    f = np.zeros((n_objects, len(scored)))
    for col, t in enumerate(scored):
        pred = labels_to_mask(soft[t], original_size, cfg["normalize_labels"]).cpu().numpy()
        truth = read_annotation(mask_paths[t])
        for obj in range(1, n_objects + 1):
            j[obj - 1, col] = region_similarity(pred == obj, truth == obj)
            f[obj - 1, col] = boundary_f_measure(pred == obj, truth == obj)

    records = []
    for obj in range(1, n_objects + 1):
        records.append(
            {
                "sequence": sequence,
                "object": obj,
                "area_fraction": float((first == obj).mean()),
                "J": float(j[obj - 1].mean()),
                "F": float(f[obj - 1].mean()),
            }
        )
    return records


def summarize(records):
    """Mean J, F and J&F over objects, overall and per size bucket."""

    def mean_of(rows):
        if not rows:
            return None
        j = float(np.mean([r["J"] for r in rows]))
        f = float(np.mean([r["F"] for r in rows]))
        return {"J&F": (j + f) / 2, "J": j, "F": f, "objects": len(rows)}

    summary = {"all": mean_of(records), "by_size": {}}
    for name, low, high in SIZE_BUCKETS:
        rows = [r for r in records if low <= r["area_fraction"] < high]
        summary["by_size"][name] = mean_of(rows)
    return summary


def resolve_config(args_eval):
    """Flatten an eval config (yaml or CLI) into the settings used above."""
    exp = args_eval.get("experiment") or {}
    data = exp.get("data") or {}
    prop = exp.get("propagation") or {}
    cfg = {
        "davis_root": data.get("davis_root"),
        "split": data.get("split", "val"),
        "short_side": data.get("short_side", 480),
        "feature_mode": exp.get("feature_mode", "frame"),
        "frames_per_clip": data.get("frames_per_clip", 16),
        "batch_size": exp.get("batch_size", 8),
        "dtype": exp.get("dtype", "float32"),
        "n_context": prop.get("n_context", 15),
        "neighborhood_size": prop.get("neighborhood_size", 12),
        "neighborhood_shape": prop.get("neighborhood_shape", "circle"),
        "topk": prop.get("topk", 5),
        "temperature": prop.get("temperature", 0.2),
        "normalize_labels": prop.get("normalize_labels", True),
    }
    if cfg["dtype"] not in DTYPES:
        raise ValueError(f"dtype must be one of {sorted(DTYPES)}, got {cfg['dtype']!r}")
    return cfg


def main(args_eval, resume_preempt=False):
    pretrain_folder = args_eval.get("folder")
    checkpoint_name = args_eval.get("checkpoint", "latest.pth.tar")
    checkpoint_key = args_eval.get("checkpoint_key", "target_encoder")
    # random_init: score an UNTRAINED encoder of the same architecture.  That
    # is the floor of this probe (label propagation gets some overlap from
    # the spatial neighbourhood alone), the number a trained arm must beat.
    random_init = args_eval.get("random_init", False)
    tag = args_eval.get("tag", "davis-tracking")
    cfg = resolve_config(args_eval)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # released: score one of Meta's released V-JEPA 2.1 encoders instead of a
    # run of ours.  There is no params-pretrain.yaml for those, so the
    # architecture comes from app/vjepa_3/pretrained.py, and results are
    # written under `pretrain_folder` (the --out folder).  This is the yardstick
    # check: does this probe put a known-good model near its published score?
    released = args_eval.get("released")
    if released is not None:
        pretrain_cfg = released_2_1_pretrain_config(released["model_name"])
    else:
        with open(os.path.join(pretrain_folder, "params-pretrain.yaml"), "r") as f:
            pretrain_cfg = yaml.safe_load(f)
    cfg["patch_size"] = pretrain_cfg["data"]["patch_size"]
    cfg["tubelet_size"] = pretrain_cfg["data"]["tubelet_size"]
    n_frames = cfg["tubelet_size"] if cfg["feature_mode"] == "frame" else cfg["frames_per_clip"]
    encoder = build_encoder(pretrain_cfg, cfg["short_side"], n_frames)

    epoch = None
    if random_init:
        logger.info("random_init: scoring an untrained encoder (probe floor)")
        checkpoint_name = "random-init"
    elif released is not None:
        checkpoint_name = os.path.basename(released["path"])
        checkpoint_key = released["checkpoint_key"]
        load_encoder_weights(
            encoder, read_encoder_state_dict(released["path"], checkpoint_key)
        )
        logger.info(f"loaded released encoder {checkpoint_name} [{checkpoint_key}]")
    else:
        checkpoint = torch.load(
            os.path.join(pretrain_folder, checkpoint_name), map_location="cpu"
        )
        load_encoder_weights(encoder, checkpoint[checkpoint_key])
        epoch = checkpoint.get("epoch")
        logger.info(f"loaded {checkpoint_key} from epoch {epoch}")
    encoder.to(device).eval()

    records, start = [], time.time()
    sequences = read_sequences(cfg["davis_root"], cfg["split"])
    for i, sequence in enumerate(sequences):
        rows = evaluate_sequence(encoder, cfg["davis_root"], sequence, device, cfg)
        records.extend(rows)
        if rows:
            jf = np.mean([(r["J"] + r["F"]) / 2 for r in rows])
            logger.info(f"[{i + 1}/{len(sequences)}] {sequence}: J&F {100 * jf:.1f}")

    report = {
        "run": pretrain_folder,
        "checkpoint": checkpoint_name,
        "epoch": epoch,
        "settings": cfg,
        "sequences": len(sequences),
        "seconds": time.time() - start,
        "summary": summarize(records),
        "per_object": records,
    }
    overall = report["summary"]["all"]
    if overall is not None:
        logger.info(
            f"DAVIS {cfg['split']} ({cfg['feature_mode']} mode): "
            f"J&F {100 * overall['J&F']:.1f}  J {100 * overall['J']:.1f}  "
            f"F {100 * overall['F']:.1f}  over {overall['objects']} objects"
        )
        for name, row in report["summary"]["by_size"].items():
            if row is not None:
                logger.info(f"  {name:6s} objects ({row['objects']:2d}): J&F {100 * row['J&F']:.1f}")

    out_dir = os.path.join(pretrain_folder, "video_tracking_frozen", tag)
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.basename(checkpoint_name).replace(".pth.tar", "").replace(".pt", "")
    out_path = os.path.join(out_dir, f"davis_{cfg['split']}_{cfg['feature_mode']}_{stem}.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"wrote {out_path}")
    return report


def _cli():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run", help="pre-training run folder")
    parser.add_argument(
        "--released",
        help="instead of --run: path of a released V-JEPA 2.1 file "
        "(e.g. vjepa2_1_vitb_dist_vitG_384.pt); needs --out",
    )
    parser.add_argument("--out", help="with --released: folder to write the result into")
    parser.add_argument("--davis", required=True, help="DAVIS root folder")
    parser.add_argument("--checkpoint", default="latest.pth.tar")
    parser.add_argument("--mode", default="frame", choices=["frame", "clip"])
    parser.add_argument("--dtype", default="float32", choices=sorted(DTYPES))
    parser.add_argument("--split", default="val")
    parser.add_argument("--random-init", action="store_true")
    args = parser.parse_args()
    if (args.run is None) == (args.released is None):
        parser.error("give exactly one of --run and --released")
    released = None
    if args.released is not None:
        if args.out is None:
            parser.error("--released needs --out")
        name = os.path.basename(args.released).replace(".pt", "")
        if name not in RELEASED_2_1:
            parser.error(
                f"{name!r} is not a released V-JEPA 2.1 file name; known: {sorted(RELEASED_2_1)}"
            )
        model_name, checkpoint_key = RELEASED_2_1[name]
        released = {
            "path": args.released,
            "model_name": model_name,
            "checkpoint_key": checkpoint_key,
        }
    return main(
        {
            "folder": args.run if released is None else args.out,
            "released": released,
            "checkpoint": args.checkpoint,
            "random_init": args.random_init,
            "experiment": {
                "feature_mode": args.mode,
                "dtype": args.dtype,
                "data": {"davis_root": args.davis, "split": args.split},
            },
        }
    )


if __name__ == "__main__":
    _cli()
