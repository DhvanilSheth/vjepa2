# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Run the REAL `train.main()` of app.vjepa_3 (or app.vjepa_2_1) on CPU.

Two things are substituted, both outside the code under test:

  * torch.distributed: a one-process gloo group, so that train.py can build
    its DistributedDataParallel wrappers without a GPU;
  * init_data: a small, finite, deterministic set of random clips, served
    through the REAL MaskCollator instance train.py constructs.

Everything else -- model build, optimizer, schedules, the loss closure, the
GradScaler calls, the EMA update, logging, checkpointing, resume -- is the
repo's own code, unmodified.

This started as the scripts the ladder audit used (V3_STRING/LADDER_AUDIT.md,
findings C-22 and L-26: "no test executes train.main").
"""

import copy
import importlib
import logging
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_CONFIG = REPO_ROOT / "configs" / "train_3" / "proxy" / "base-1gpu.yaml"


def init_gloo():
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", str(29500 + os.getpid() % 2000))
        dist.init_process_group(backend="gloo", world_size=1, rank=0)


def deep_update(target, updates):
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            deep_update(target[key], value)
        else:
            target[key] = value
    return target


def tiny_config(folder, model_name="vit_tiny", epochs=2, ipe=3, **overrides):
    """The proxy recipe shrunk to something a laptop CPU trains in seconds.

    `overrides` is a nested dict merged on top, e.g.
    `loss={"reg_type": "sigreg"}`.
    """
    cfg = yaml.safe_load(BASE_CONFIG.read_text())
    cfg["folder"] = str(folder)
    cfg["data"].update(
        batch_size=4, crop_size=64, dataset_fpcs=[8], num_workers=0, pin_mem=False
    )
    cfg["model"].update(
        model_name=model_name, pred_depth=4, pred_embed_dim=96, pred_num_heads=3
    )
    cfg["optimization"].update(epochs=epochs, ipe=ipe, warmup=1)
    cfg["meta"].update(dtype="float32", sigreg_diag_freq=0, save_every_freq=1)
    return deep_update(cfg, copy.deepcopy(overrides))


class SyntheticClips(torch.utils.data.Dataset):
    """Same item layout as src/datasets/video_dataset.py: ([clip], label, [indices]).

    kind="noise": independent Gaussian pixels (the default; fastest).
    kind="smooth": a low-resolution random image, upsampled and shifted a few
    pixels per frame -- large flat regions and frame-to-frame coherence, which
    is closer to video than white noise is.  Used by scripts/preflight_arm.py.
    """

    def __init__(self, n, frames, crop, kind="noise"):
        self.n, self.frames, self.crop, self.kind = n, frames, crop, kind

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        gen = torch.Generator().manual_seed(1000 + index)
        if self.kind == "noise":
            clip = torch.randn(3, self.frames, self.crop, self.crop, generator=gen)
        else:
            coarse = torch.randn(1, 3, 6, 6, generator=gen)
            image = torch.nn.functional.interpolate(
                coarse, size=(self.crop, self.crop), mode="bicubic", align_corners=False
            )[0]
            dx, dy = (int(v) for v in torch.randint(-3, 4, (2,), generator=gen))
            frames = [torch.roll(image, (t * dy, t * dx), dims=(1, 2)) for t in range(self.frames)]
            clip = torch.stack(frames, dim=1)
        return [clip], 0, [np.arange(self.frames, dtype=np.int64)]


class EpochSampler(torch.utils.data.Sampler):
    def __init__(self, n):
        self.n, self.epoch, self.calls = n, 0, []

    def set_epoch(self, epoch):
        self.epoch = epoch
        self.calls.append(epoch)

    def __len__(self):
        return self.n

    def __iter__(self):
        order = np.random.default_rng(self.epoch).permutation(self.n)
        return iter(order.tolist())


def run_train(
    cfg, app="vjepa_3", n_clips=12, before_step=None, on_optimizer_step=None, clip_kind="noise"
):
    """Run train.main(cfg) to completion and return what it left on disk.

    `before_step(global_step)` is called just before each training step; raise
    from it to simulate a crash.  `on_optimizer_step(encoder, predictor)` is
    called inside each optimizer step, when the gradients are in place.
    Returns {"latest": checkpoint dict, "sampler": the sampler (for its
    set_epoch calls), "init_data_kwargs": what train.py passed to init_data,
    "init_encoder" / "init_predictor": the weights before the first step}.
    """
    init_gloo()
    train = importlib.import_module(f"app.{app}.train")
    captured = {}
    state = {"step": 0}

    def fake_init_data(batch_size, collator=None, dataset_fpcs=None, **kwargs):
        captured["init_data_kwargs"] = dict(kwargs, batch_size=batch_size)
        dataset = SyntheticClips(
            n_clips, dataset_fpcs[0], cfg["data"]["crop_size"], kind=clip_kind
        )
        sampler = EpochSampler(n_clips)
        loader = torch.utils.data.DataLoader(
            dataset,
            collate_fn=collator,
            sampler=sampler,
            batch_size=batch_size,
            drop_last=True,
            num_workers=0,
        )
        captured["sampler"] = sampler
        return loader, sampler

    real_init_data, real_timer = train.init_data, train.gpu_timer
    real_init_model, real_adamw_step = train.init_video_model, torch.optim.AdamW.step

    def init_model(**kwargs):
        encoder, predictor = real_init_model(**kwargs)
        captured["models"] = (encoder, predictor)
        return encoder, predictor

    def adamw_step(optimizer, *args, **kwargs):
        if "init_encoder" not in captured:
            encoder, predictor = captured["models"]
            captured["init_encoder"] = copy.deepcopy(encoder.state_dict())
            captured["init_predictor"] = copy.deepcopy(predictor.state_dict())
        if on_optimizer_step is not None:
            on_optimizer_step(*captured["models"])
        return real_adamw_step(optimizer, *args, **kwargs)

    def timer(closure, **kwargs):
        if before_step is not None:
            before_step(state["step"])
        out = real_timer(closure, **kwargs)
        state["step"] += 1
        return out

    # train.py seeds python's `random` once, at import.  Several runs share
    # this interpreter, so re-seed to emulate a fresh process each time.
    random.seed(0)
    os.makedirs(cfg["folder"], exist_ok=True)
    train.init_data, train.gpu_timer = fake_init_data, timer
    train.init_video_model, torch.optim.AdamW.step = init_model, adamw_step
    logging.disable(logging.CRITICAL)
    try:
        train.main(args=copy.deepcopy(cfg))
    finally:
        train.init_data, train.gpu_timer = real_init_data, real_timer
        train.init_video_model, torch.optim.AdamW.step = real_init_model, real_adamw_step
        logging.disable(logging.NOTSET)
    captured.pop("models", None)

    latest = os.path.join(cfg["folder"], "latest.pth.tar")
    if os.path.exists(latest):
        captured["latest"] = torch.load(latest, map_location="cpu", weights_only=False)
    captured["steps"] = state["step"]
    return captured


def states_equal(a, b):
    """Bitwise equality of two state dicts (same keys, same tensors)."""
    if a.keys() != b.keys():
        return False
    return all(torch.equal(a[k], b[k]) for k in a)


def max_abs_diff(a, b):
    return max(float((a[k].float() - b[k].float()).abs().max()) for k in a)
