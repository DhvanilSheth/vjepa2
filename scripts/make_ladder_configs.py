# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Generate the V-JEPA 3 ablation ladder from one base config.

    python3 scripts/make_ladder_configs.py            # (re)write the arm files
    python3 scripts/make_ladder_configs.py --check    # exit 1 if they are stale

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

# name -> (what it tests, warm-startable from a 2.1 checkpoint, model overrides)
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


def _set_folder(text, name):
    pattern = re.compile(r"^folder:.*$", re.M)
    assert len(pattern.findall(text)) == 1
    return pattern.sub(f"folder: {FOLDER_ROOT}/{name}", text)


def render(name, description, warm_start, overrides):
    """Return the YAML text for one arm."""
    base_text = BASE.read_text()
    text = _set_folder(base_text, name)
    for key, value in overrides.items():
        text = _set_key(text, "model", key, value)

    # The rewrite is textual (to keep the base file's comments); prove it did
    # exactly what the table says by comparing parsed configs.
    expected = yaml.safe_load(base_text)
    expected["folder"] = f"{FOLDER_ROOT}/{name}"
    expected["model"].update(overrides)
    assert yaml.safe_load(text) == expected, f"{name}: rendered config != base + overrides"

    changed = ", ".join(f"{k}: {_yaml_scalar(v)}" for k, v in overrides.items()) or "nothing"
    header = (
        "# GENERATED by scripts/make_ladder_configs.py from\n"
        f"# configs/train_3/proxy/{BASE.name} -- do not edit this file.\n"
        "#\n"
        f"# ARM {name}: {description}\n"
        f"# Changes vs a0-control: {changed}\n"
        f"# Warm start from a V-JEPA 2.1 checkpoint: {'yes' if warm_start else 'NO - train from scratch'}\n"
        "#\n"
        "# Everything below this line is the base file with only those keys (and\n"
        "# `folder`) rewritten; the comments are the base file's own.\n"
        "#\n"
    )
    return header + text


def build_all():
    """{output path: text} for every arm, plus the preset when one is defined."""
    files = {
        OUT_DIR / f"{name}.yaml": render(name, *spec) for name, spec in ARMS.items()
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
        files[OUT_DIR / "modded.yaml"] = render(
            "modded", "Preset: " + " + ".join(PRESET) + ".", warm, merged
        )
    return files


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--check", action="store_true", help="verify the files on disk; write nothing"
    )
    args = parser.parse_args(argv)

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


if __name__ == "__main__":
    sys.exit(main())
