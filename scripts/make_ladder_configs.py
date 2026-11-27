# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Generate the V-JEPA 3 ablation ladder from one base config.

    python3 scripts/make_ladder_configs.py            # (re)write the arm files
    python3 scripts/make_ladder_configs.py --check    # exit 1 if they are stale

    # On the training machine: write copies with real paths to a folder that
    # is NOT tracked, instead of editing the files above by hand.
    python3 scripts/make_ladder_configs.py --local configs/train_3/local \
        --folder-root /runs/pretrain_3/ladder --data /data/train_paths.csv \
        [--pretrained /ckpt/vjepa2_1_vitb_dist_vitG_384.pt] [--seed 1239]

WHY A GENERATOR
---------------
An ablation is only readable if each arm differs from the control in exactly
the keys it claims to.  Hand-copied 150-line YAML files drift.  Here every arm
is the base file with a few `model:` lines rewritten, the ARMS table below is
the single statement of what each arm changes, and
tests/models/test_vjepa3_configs.py checks that the files on disk are exactly
what this script produces.

THE LADDER
----------
One change per arm, each compared against `a0-control`.  "warm start" says
whether the arm can be initialised from a V-JEPA 2.1 checkpoint: positional
changes can (identical parameters, and STRING starts at the identity), while a
new FFN or norm changes the function the weights compute, so those train from
scratch -- and must be compared against a from-scratch control.

AFTER THE LADDER: THE PRESET
----------------------------
List the winning arms in PRESET and re-run; `modded.yaml` is then written as
the union of their overrides.  It is empty on purpose until there are results:
a preset built from guesses cannot be attributed to anything.

OVERRIDE KEYS
-------------
A plain key ("rope_theta") is a `model:` key.  A dotted key
("loss.reg_type") names its section.  A key with a number
("mask.0.max_temporal_keep") names one entry of a list section, counted from
0.  Every key must already exist in the base file at its baseline value.

FOUR FAMILIES
-------------
  a / p / f / n   the first screen: architecture changes, from scratch,
                  compared with a0-control.
  s               regularisers added to the loss, from scratch, compared with
                  a0-control.
  t               how the video is fed and what is hidden (time-focused),
                  from scratch, compared with a0-control.
  w               encoder started from Meta's released V-JEPA 2.1 ViT-B
                  (app/vjepa_3/pretrained.py).  Compared with w0-warm-control,
                  NOT with a0-control: the starting point differs.
"""

import argparse
import re
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
BASE = REPO_ROOT / "configs" / "train_3" / "proxy" / "base-1gpu.yaml"
OUT_DIR = REPO_ROOT / "configs" / "train_3" / "ladder"
FOLDER_ROOT = "/your_folder/pretrain_3/ladder"
DATA_PLACEHOLDER = "/your_data/ssv2_train_paths.csv"
PRETRAINED_PLACEHOLDER = "/your_checkpoints/vjepa2_1_vitb_dist_vitG_384.pt"

# Regulariser weights for the s-arms.  FIRST GUESSES, not tuned values.
#
# The papers weight their regulariser about 10x the prediction loss, but there
# it is the only thing preventing collapse; here the EMA teacher stays, so the
# regulariser should push, not take over.  These two values were set with
# scripts/preflight_arm.py (ViT-B, one step, synthetic clips, Nov 2026): at
# weight 0.5 SIGReg added a gradient 1.8x the main loss's on the encoder and
# VISReg 3.3x.  The weights below bring both to roughly a third of the main
# loss's gradient at the first step, so the two arms push about equally hard.
# Real video and later steps will differ.
#
# Check <folder>/reg_r0.csv after the first epoch (300 iterations): if
# loss_reg has not fallen, the weight is too small to matter; if loss_main
# sits well above the control's first-epoch loss, it is too large.
SIGREG_WEIGHT = 0.1
VISREG_WEIGHT = 0.05

# The warm-start guards shared by every w-arm (see train.py).  Also first
# guesses: the predictor trains alone for the lr warm-up (900 iterations),
# then the encoder moves at a tenth of the schedule's learning rate.
WARM_START = {
    "meta.pretrained_encoder": PRETRAINED_PLACEHOLDER,
    "optimization.freeze_encoder_iters": 900,
    "optimization.encoder_lr_scale": 0.1,
}

# -- Time-focused changes to WHAT THE MODEL IS ASKED and HOW VIDEO IS SAMPLED.
#    Neither touches the network, so both also exist in the warm-start family.
#
# FUTURE: the recipe has two masks per clip.  Both hide the same spatial
# blocks in EVERY frame, so nothing in the task requires following an object
# through time.  Here mask 0 (eight small blocks; about a quarter of the clip
# stays visible) keeps its context to the first half of the clip: the encoder
# sees frames 1-8 and the predictor must produce all of frames 9-16.  Mask 1
# (two large blocks) is unchanged, so the encoder still sees every time
# position in half of its passes.
FUTURE = {"mask.0.max_temporal_keep": 0.5}
#
# FPS12: frames are taken 12 per second instead of 4, so the 16 frames span
# 1.3 s instead of 4 s and neighbouring frames show small displacements.  The
# tracking test runs on consecutive 24-fps frames; at 4 fps the model never
# trains on motion that small.
FPS12 = {"data.fps": 12}

#
# FRAMES: mask 0 hides whole moments instead of regions: 6 of the 8 time-steps,
# chosen at random per clip, are hidden completely and predicted from the 2
# that are shown (app/vjepa_3/masks.py).  6 was chosen to MATCH THE CONTROL'S
# BUDGET: 2 visible time-steps are 512 tokens, the control's mask 0 shows
# about 504.  So the arm changes how the visible quarter is arranged (two
# whole frames against the same patches in every frame), not how much is
# shown.  Mask 1 is unchanged.
FRAMES = {"mask.0.hide_time_steps": 6}
#
# WHOLE_CLIP: both masks keep a random subset when a batch's token lists are
# cut to a common length, instead of 2.1's cut from the end.  Same number of
# tokens shown and asked about; they are spread over the whole clip instead
# of leaving its last quarter largely unused.
WHOLE_CLIP = {"mask.0.truncation": "random", "mask.1.truncation": "random"}

# name -> (what it tests, can the encoder be loaded from a 2.1 checkpoint, overrides)
ARMS = {
    "a0-control": (
        "Control. V-JEPA 2.1 behaviour on the proxy recipe.",
        True,
        {},
    ),
    "p1-theta100": (
        "RoPE base 100 on all axes (more rotary channels live at 8-32 positions).",
        True,
        {"rope_theta": 100.0},
    ),
    "p2-theta-temporal": (
        "RoPE base 100 on the temporal axis only; isolates the temporal effect of p1.",
        True,
        {"rope_theta": [100.0, 10000.0, 10000.0]},
    ),
    "p3-string": (
        "Cayley-STRING learned basis on the encoder (starts at the identity).",
        True,
        {"use_string": True},
    ),
    "f1-swiglu": (
        "SwiGLU feed-forward in encoder and predictor.",
        False,
        {"ffn_type": "swiglu", "pred_ffn_type": "swiglu"},
    ),
    "f2-relu2": (
        "Squared-ReLU MLP in encoder and predictor (same parameters as GELU).",
        False,
        {"ffn_type": "relu2", "pred_ffn_type": "relu2"},
    ),
    "n1-dyt": (
        "Dynamic Tanh in place of every LayerNorm.",
        False,
        {"norm_type": "dyt"},
    ),
    "n2-derf": (
        "Dynamic erf (Derf) in place of every LayerNorm.",
        False,
        {"norm_type": "derf"},
    ),
    "s1-sigreg": (
        "SIGReg (LeJEPA) on the encoder's output tokens, added to the loss.",
        True,
        {"loss.reg_type": "sigreg", "loss.reg_weight": SIGREG_WEIGHT},
    ),
    "s2-visreg": (
        "VISReg (centre + scale + sliced-Wasserstein shape) on the encoder's output tokens.",
        True,
        {"loss.reg_type": "visreg", "loss.reg_weight": VISREG_WEIGHT},
    ),
    "n3-dyt-sigreg": (
        "n1-dyt with SIGReg: tests whether an explicit anti-collapse term can stand in for LayerNorm.",
        False,
        {"norm_type": "dyt", "loss.reg_type": "sigreg", "loss.reg_weight": SIGREG_WEIGHT},
    ),
    "n4-derf-sigreg": (
        "n2-derf with SIGReg. Compare with n3-dyt-sigreg and s1-sigreg.",
        False,
        {"norm_type": "derf", "loss.reg_type": "sigreg", "loss.reg_weight": SIGREG_WEIGHT},
    ),
    "t1-future": (
        "Future prediction on mask 0: context from the first half of the clip only.",
        True,
        dict(FUTURE),
    ),
    "t2-fps12": (
        "Frames sampled at 12 fps instead of 4 (16 frames span 1.3 s instead of 4 s).",
        True,
        dict(FPS12),
    ),
    "t3-frames": (
        "Mask 0 hides 6 of 8 whole time-steps instead of spatial blocks.",
        True,
        dict(FRAMES),
    ),
    "t4-whole-clip": (
        "Token lists are cut to a common length at random, not from the end of the clip.",
        True,
        dict(WHOLE_CLIP),
    ),
    "w0-warm-control": (
        "Control of the warm-start family: released V-JEPA 2.1 ViT-B encoder, nothing else changed.",
        True,
        dict(WARM_START),
    ),
    "w2-warm-theta-temporal": (
        "p2-theta-temporal on top of the released encoder. Compare with w0-warm-control.",
        True,
        dict(WARM_START, rope_theta=[100.0, 10000.0, 10000.0]),
    ),
    "w3-warm-future": (
        "t1-future on top of the released encoder. Compare with w0-warm-control.",
        True,
        dict(WARM_START, **FUTURE),
    ),
    "w4-warm-fps12": (
        "t2-fps12 on top of the released encoder. Compare with w0-warm-control.",
        True,
        dict(WARM_START, **FPS12),
    ),
    "w5-warm-frames": (
        "t3-frames on top of the released encoder. Compare with w0-warm-control.",
        True,
        dict(WARM_START, **FRAMES),
    ),
    "w6-warm-whole-clip": (
        "t4-whole-clip on top of the released encoder. Compare with w0-warm-control.",
        True,
        dict(WARM_START, **WHOLE_CLIP),
    ),
}

# Arms to combine into configs/train_3/ladder/modded.yaml.  Fill in from
# results, e.g. ["p2-theta-temporal", "n2-derf"].
PRESET = []


def _yaml_scalar(value):
    """One-line YAML for an override value (lists in flow style)."""
    return yaml.safe_dump(value, default_flow_style=True).strip().removesuffix("\n...").strip()


def _set_key(text, section, key, value):
    """Rewrite the single `  key: ...` line inside top-level `section:`."""
    start = text.index(f"\n{section}:\n") + 1
    match = re.compile(r"^\S", re.M).search(text, start + len(section) + 2)
    end = match.start() if match else len(text)
    body = text[start:end]
    pattern = re.compile(rf"^  {re.escape(key)}:.*$", re.M)
    hits = pattern.findall(body)
    if len(hits) != 1:
        raise KeyError(
            f"expected exactly one `{key}:` line under `{section}:` in the base "
            f"config, found {len(hits)}. Add the key to {BASE.name} at its "
            "baseline value first."
        )
    body = pattern.sub(lambda _: f"  {key}: {_yaml_scalar(value)}", body)
    return text[:start] + body + text[end:]


def _set_list_key(text, section, index, key, value):
    """Rewrite `key` inside entry `index` of the top-level list `section:`.

    The list is block-style: each entry starts with "- " in column 0 and its
    other keys are indented by two spaces.
    """
    start = text.index(f"\n{section}:\n") + len(section) + 3
    match = re.compile(r"^[^\s-]", re.M).search(text, start)
    end = match.start() if match else len(text)
    body = text[start:end]
    starts = [m.start() for m in re.finditer(r"^- ", body, re.M)]
    if index >= len(starts):
        raise KeyError(f"`{section}:` has {len(starts)} entries; there is no entry {index}")
    stop = starts[index + 1] if index + 1 < len(starts) else len(body)
    item = body[starts[index] : stop]
    pattern = re.compile(rf"^(  |- ){re.escape(key)}:.*$", re.M)
    hits = pattern.findall(item)
    if len(hits) != 1:
        raise KeyError(
            f"expected exactly one `{key}:` line in entry {index} of `{section}:` "
            f"in the base config, found {len(hits)}."
        )
    item = pattern.sub(lambda m: f"{m.group(1)}{key}: {_yaml_scalar(value)}", item)
    body = body[: starts[index]] + item + body[stop:]
    return text[:start] + body + text[end:]


def _set_folder(text, name, folder_root):
    pattern = re.compile(r"^folder:.*$", re.M)
    assert len(pattern.findall(text)) == 1
    return pattern.sub(lambda _: f"folder: {folder_root}/{name}", text)


def split_key(key):
    """Override key -> (section, list index or None, key).

    "rope_theta" -> ("model", None, "rope_theta");
    "loss.reg_type" -> ("loss", None, "reg_type");
    "mask.0.max_temporal_keep" -> ("mask", 0, "max_temporal_keep").
    """
    parts = key.split(".")
    if len(parts) == 1:
        return "model", None, parts[0]
    if len(parts) == 2:
        return parts[0], None, parts[1]
    if len(parts) == 3 and parts[1].isdigit():
        return parts[0], int(parts[1]), parts[2]
    raise KeyError(f"cannot read override key {key!r}")


def render(name, description, warm_start, overrides, local=None):
    """Return the YAML text for one arm.

    `local` (optional) turns the tracked, placeholder-path file into a copy for
    one machine: {"folder_root", "data", "pretrained", "seed", "suffix"}.
    """
    local = local or {}
    run_name = name + local.get("suffix", "")
    folder_root = local.get("folder_root") or FOLDER_ROOT
    overrides = dict(overrides)
    if local.get("seed") is not None:
        overrides["meta.seed"] = local["seed"]
    if local.get("pretrained") and "meta.pretrained_encoder" in overrides:
        overrides["meta.pretrained_encoder"] = local["pretrained"]

    base_text = BASE.read_text()
    if local.get("data"):
        assert base_text.count(DATA_PLACEHOLDER) == 1, "data placeholder not found in base"
        base_text = base_text.replace(DATA_PLACEHOLDER, local["data"])
    text = _set_folder(base_text, run_name, folder_root)
    for key, value in overrides.items():
        section, index, plain = split_key(key)
        if index is None:
            text = _set_key(text, section, plain, value)
        else:
            text = _set_list_key(text, section, index, plain, value)

    # The rewrite is textual (to keep the base file's comments); prove it did
    # exactly what the table says by comparing parsed configs.
    expected = yaml.safe_load(base_text)
    expected["folder"] = f"{folder_root}/{run_name}"
    for key, value in overrides.items():
        section, index, plain = split_key(key)
        target = expected[section] if index is None else expected[section][index]
        assert plain in target, f"{name}: `{key}` is not in the base config"
        target[plain] = value
    assert yaml.safe_load(text) == expected, f"{name}: rendered config != base + overrides"

    changed = ", ".join(f"{k}: {_yaml_scalar(v)}" for k, v in overrides.items()) or "nothing"
    local_note = (
        "# LOCAL COPY for one machine (real paths); regenerate, do not commit.\n" if local else ""
    )
    header = (
        "# GENERATED by scripts/make_ladder_configs.py from\n"
        f"# configs/train_3/proxy/{BASE.name} -- do not edit this file.\n"
        f"{local_note}"
        "#\n"
        f"# ARM {run_name}: {description}\n"
        f"# Changes vs the base file: {changed}\n"
        "# Encoder can be loaded from a V-JEPA 2.1 checkpoint: "
        f"{'yes' if warm_start else 'NO - different parameters'}"
        " (only the w-arms actually do; every other arm trains from scratch)\n"
        "#\n"
        "# Everything below this line is the base file with only those keys (and\n"
        "# `folder`) rewritten; the comments are the base file's own.\n"
        "#\n"
    )
    return header + text


def build_all(out_dir=None, local=None):
    """{output path: text} for every arm, plus the preset when one is defined."""
    out_dir = OUT_DIR if out_dir is None else Path(out_dir)
    suffix = (local or {}).get("suffix", "")
    files = {
        out_dir / f"{name}{suffix}.yaml": render(name, *spec, local=local)
        for name, spec in ARMS.items()
    }
    if PRESET:
        unknown = [arm for arm in PRESET if arm not in ARMS]
        if unknown:
            raise KeyError(f"PRESET names unknown arms: {unknown}")
        merged = {}
        for arm in PRESET:
            for key, value in ARMS[arm][2].items():
                if key in merged and merged[key] != value:
                    raise ValueError(
                        f"PRESET arms disagree on `{key}`: {merged[key]!r} vs {value!r}"
                    )
                merged[key] = value
        warm = all(ARMS[arm][1] for arm in PRESET)
        files[out_dir / f"modded{suffix}.yaml"] = render(
            "modded", "Preset: " + " + ".join(PRESET) + ".", warm, merged, local=local
        )
    return files


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--check", action="store_true", help="verify the files on disk; write nothing"
    )
    parser.add_argument(
        "--local",
        metavar="DIR",
        help="write copies with this machine's paths to DIR (never the tracked folder)",
    )
    parser.add_argument("--folder-root", help="--local: parent folder of the run folders")
    parser.add_argument("--data", help="--local: path of the training video list (csv)")
    parser.add_argument(
        "--pretrained", help="--local: path of the released encoder file, for the w-arms"
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="--local: set meta.seed and add -seed<N> to file and folder names",
    )
    args = parser.parse_args(argv)

    if args.local:
        return _write_local(args)
    if args.folder_root or args.data or args.pretrained or args.seed is not None:
        parser.error("--folder-root, --data, --pretrained and --seed need --local DIR")

    files = build_all()
    expected_names = {path.name for path in files}
    stale = [
        path for path, text in files.items()
        if not path.exists() or path.read_text() != text
    ]
    extra = sorted(
        p for p in OUT_DIR.glob("*.yaml") if p.name not in expected_names
    ) if OUT_DIR.is_dir() else []

    if args.check:
        for path in stale:
            print(f"STALE   {path.relative_to(REPO_ROOT)}")
        for path in extra:
            print(f"ORPHAN  {path.relative_to(REPO_ROOT)}  (no such arm)")
        if stale or extra:
            print("Run: python3 scripts/make_ladder_configs.py")
            return 1
        print(f"ok: {len(files)} ladder config(s) match the generator")
        return 0

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for path, text in files.items():
        path.write_text(text)
        print(f"wrote   {path.relative_to(REPO_ROOT)}")
    for path in extra:
        print(f"ORPHAN  {path.relative_to(REPO_ROOT)}  (no such arm; delete it by hand)")
    return 0


def _write_local(args):
    out_dir = Path(args.local).resolve()
    if out_dir == OUT_DIR.resolve():
        print(f"refusing to write local copies into the tracked folder {OUT_DIR}")
        return 1
    if not args.folder_root or not args.data:
        print("--local needs --folder-root and --data")
        return 1
    local = {
        "folder_root": args.folder_root.rstrip("/"),
        "data": args.data,
        "pretrained": args.pretrained,
        "seed": args.seed,
        "suffix": "" if args.seed is None else f"-seed{args.seed}",
    }
    files = build_all(out_dir=out_dir, local=local)
    out_dir.mkdir(parents=True, exist_ok=True)
    for path, text in files.items():
        placeholder = PRETRAINED_PLACEHOLDER in text
        path.write_text(text)
        print(f"wrote   {path}" + ("   (needs --pretrained before it can run)" if placeholder else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
