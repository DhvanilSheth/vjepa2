# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Frozen-feature clustering probe for V-JEPA 3 checkpoints.

WHAT IT MEASURES
----------------
Freeze the encoder, embed every clip of a labelled video set as one vector
(mean over patch tokens), cluster those vectors WITHOUT labels, then ask how
well the clusters line up with the classes (purity, NMI).  No probe is
trained, so there are no probe hyperparameters to tune per arm -- which is
what makes it a cheap, fair first reading across an ablation ladder.  It also
reports the SIGReg isotropy reading (app/vjepa_3/diagnostics.py) of the same
features.

It is a COARSE instrument: mean-pooling discards where and when things
happen, so this mostly rewards appearance/scene semantics.  It does not
replace the attentive-probe benchmarks, and says little about motion.

WHY IT EXISTS SEPARATELY FROM THE OTHER EVALS
---------------------------------------------
The other eval entry points build the encoder from `src.models`, whose RoPE
tiles frequencies instead of interleaving them (see V3_STRING/README.md §6).
This one builds from `app.vjepa_3.models`, and reads the architecture flags
from the run's own `params-pretrain.yaml`, so a checkpoint is always loaded
into exactly the model that produced it (`strict=True`).

RUN (single GPU)
----------------
    python -m evals.main --fname configs/eval_3/vitb16/cluster-probe.yaml \\
        --devices cuda:0

STATUS
------
The pure pieces (kwargs mapping, weight loading, pooling, clustering,
metrics) are unit-tested in tests/evals/test_cluster_probe.py.  `main()` --
the dataloader and real-checkpoint path -- has not been executed: it needs
decord, a dataset and a trained checkpoint.
"""

import json
import os
import time

import torch
import yaml

import app.vjepa_3.models.vision_transformer as vit_3
from app.vjepa_3.diagnostics import gaussian_floor, isotropy_report
from evals.feature_clustering_frozen.cluster import CLUSTERERS, quantization_error
from evals.feature_clustering_frozen.metrics import nmi, purity
from src.utils.logging import get_logger

logger = get_logger("feature_clustering_frozen")


def encoder_kwargs_from_pretrain(cfg, resolution, frames_per_clip):
    """Map a pre-training config to the encoder factory's keyword arguments.

    Mirrors the encoder call in `app/vjepa_3/utils.py:init_video_model`, with
    the SAME defaults `app/vjepa_3/train.py` uses when a key is absent.  The
    two lists are kept in step by
    `test_probe_covers_every_encoder_kwarg_of_init_video_model`, which fails
    if a new architecture kwarg is added there and not here.

    `resolution` / `frames_per_clip` come from the probe config, not the
    pre-training one: RoPE is size-agnostic, so the encoder can be evaluated
    at a different clip geometry than it was trained at.
    """
    model = cfg["model"]
    data = cfg["data"]
    meta = cfg.get("meta") or {}
    return dict(
        img_size=resolution,
        patch_size=data["patch_size"],
        num_frames=frames_per_clip,
        tubelet_size=data["tubelet_size"],
        uniform_power=model.get("uniform_power", False),
        use_sdpa=meta.get("use_sdpa", False),
        use_silu=model.get("use_silu", False),
        wide_silu=model.get("wide_silu", True),
        # inference only: checkpointing saves memory for backward, which
        # never runs here
        use_activation_checkpointing=False,
        is_causal=model.get("is_causal", False),
        use_rope=model.get("use_rope", False),
        init_type=model.get("init_type", "default"),
        img_temporal_dim_size=model.get("img_temporal_dim_size", None),
        n_registers=model.get("n_registers", 0),
        has_cls_first=model.get("has_cls_first", False),
        interpolate_rope=model.get("interpolate_rope", False),
        rope_theta=model.get("rope_theta", 10000.0),
        rope_temporal_scale=model.get("rope_temporal_scale", 1.0),
        use_string=model.get("use_string", False),
        string_share_across_layers=model.get("string_share_across_layers", False),
        string_init_std=model.get("string_init_std", 0.0),
        modality_embedding=model.get("modality_embedding", False),
        ffn_type=model.get("ffn_type", None),
        norm_type=model.get("norm_type", None),
        norm_out_type=model.get("norm_out_type", None),
    )


def build_encoder(cfg, resolution, frames_per_clip):
    model_name = cfg["model"]["model_name"]
    kwargs = encoder_kwargs_from_pretrain(cfg, resolution, frames_per_clip)
    return vit_3.__dict__[model_name](**kwargs)


def load_encoder_weights(encoder, state_dict):
    """Load a train.py checkpoint entry into a bare encoder, strictly.

    train.py saves `DistributedDataParallel(MultiSeqWrapper(vit))`, so every
    key carries a `module.backbone.` prefix.  strict=True on purpose: the
    older eval loader uses strict=False and merely logs missing keys, which
    is how an architecture mismatch turns into a quietly random-initialised
    layer and a wrong number.
    """
    cleaned = {}
    for key, value in state_dict.items():
        for prefix in ("module.", "backbone."):
            if key.startswith(prefix):
                key = key[len(prefix) :]
        cleaned[key] = value
    encoder.load_state_dict(cleaned, strict=True)
    return encoder


@torch.no_grad()
def pool_clip_features(encoder, clips):
    """One vector per video: mean over tokens, then over segments and views.

    `clips` is the eval dataloader's nesting: a list over temporal segments,
    each a list over spatial views, each a [B, C, T, H, W] tensor.
    Returns [B, D] float32.
    """
    pooled = [encoder(view).float().mean(dim=1) for segment in clips for view in segment]
    return torch.stack(pooled).mean(dim=0)


@torch.no_grad()
def extract_features(encoder, loader, device, use_bfloat16=False):
    feats, labels = [], []
    for itr, data in enumerate(loader):
        clips = [[view.to(device, non_blocking=True) for view in seg] for seg in data[0]]
        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=use_bfloat16
        ):
            feats.append(pool_clip_features(encoder, clips).cpu())
        labels.append(data[1].cpu())
        if itr % 20 == 0:
            logger.info(f"embedded {sum(len(f) for f in feats)} clips")
    return torch.cat(feats), torch.cat(labels)


def score_clusterings(feats, labels, clusterers, k, seed=0):
    """Cluster `feats` with each named clusterer and score against `labels`."""
    results = {}
    classes = labels.numpy()
    for name in clusterers:
        if name not in CLUSTERERS:
            raise ValueError(f"Unknown clusterer {name!r}. Known: {sorted(CLUSTERERS)}")
        start = time.time()
        centroids, assigned = CLUSTERERS[name](feats, k, seed=seed)
        elapsed = time.time() - start
        results[name] = {
            "purity": purity(assigned.cpu().numpy(), classes),
            "nmi": nmi(assigned.cpu().numpy(), classes),
            "quantization_error": quantization_error(feats, centroids, assigned),
            "seconds": elapsed,
            "k": int(k),
        }
    return results


def main(args_eval, resume_preempt=False):
    pretrain_folder = args_eval.get("folder")
    checkpoint_name = args_eval.get("checkpoint", "latest.pth.tar")
    checkpoint_key = args_eval.get("checkpoint_key", "target_encoder")
    tag = args_eval.get("tag", "cluster-probe")

    args_exp = args_eval.get("experiment")
    args_data = args_exp.get("data")
    args_cluster = args_exp.get("clustering") or {}
    resolution = args_data.get("resolution", 256)
    frames_per_clip = args_data.get("frames_per_clip", 16)
    num_classes = args_data.get("num_classes")
    clusterers = args_cluster.get("clusterers", ["kmeans"])
    k = args_cluster.get("k") or num_classes
    seed = args_cluster.get("seed", 0)
    use_bfloat16 = args_exp.get("use_bfloat16", False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    with open(os.path.join(pretrain_folder, "params-pretrain.yaml"), "r") as f:
        pretrain_cfg = yaml.safe_load(f)
    encoder = build_encoder(pretrain_cfg, resolution, frames_per_clip)
    checkpoint = torch.load(
        os.path.join(pretrain_folder, checkpoint_name), map_location="cpu"
    )
    load_encoder_weights(encoder, checkpoint[checkpoint_key])
    logger.info(f"loaded {checkpoint_key} from epoch {checkpoint.get('epoch')}")
    encoder.to(device).eval()

    # Imported here so the pure functions above stay importable (and
    # testable) on a machine without decord.
    from evals.video_classification_frozen.eval import make_dataloader

    loader, _ = make_dataloader(
        root_path=[args_data.get("dataset_val")],
        batch_size=args_exp.get("batch_size", 16),
        world_size=1,
        rank=0,
        dataset_type=args_data.get("dataset_type", "VideoDataset"),
        img_size=resolution,
        frames_per_clip=frames_per_clip,
        frame_step=args_data.get("frame_step", 4),
        num_segments=args_data.get("num_segments", 1),
        eval_duration=args_data.get("clip_duration", None),
        num_views_per_segment=args_data.get("num_views_per_segment", 1),
        training=False,
        num_workers=args_exp.get("num_workers", 8),
        normalization=args_data.get("normalization", None),
    )
    feats, labels = extract_features(encoder, loader, device, use_bfloat16)
    logger.info(f"features: {tuple(feats.shape)}, classes: {labels.unique().numel()}")

    report = {
        "checkpoint": os.path.join(pretrain_folder, checkpoint_name),
        "epoch": checkpoint.get("epoch"),
        "num_clips": int(feats.size(0)),
        "isotropy": isotropy_report(feats),
        "isotropy_floor": {
            "sigreg_raw": gaussian_floor(),
            "sigreg_std": gaussian_floor(standardized=True),
        },
        "clustering": score_clusterings(feats, labels, clusterers, k, seed=seed),
    }
    logger.info(json.dumps(report, indent=2))

    out_dir = os.path.join(pretrain_folder, "feature_clustering_frozen", tag)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "cluster_probe.json"), "w") as f:
        json.dump(report, f, indent=2)
    return report
