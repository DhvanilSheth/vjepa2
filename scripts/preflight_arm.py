# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Pre-flight check for a ladder arm: one REAL training step, compared with the control.

    python3 scripts/preflight_arm.py configs/train_3/ladder/s1-sigreg.yaml
    python3 scripts/preflight_arm.py configs/train_3/ladder/*.yaml     # all arms

WHY
---
The first ladder spent five GPU-hours on `n1-dyt` before it was clear the arm
could not train: with DyT in place of LayerNorm the encoder's gradient at the
first step is thousands of times smaller than the control's
(V3_STRING/LADDER_AUDIT.md, N-09 to N-11).  That is visible in ONE step, on a
laptop CPU, in under a minute.  Run this before any new arm goes to the GPU.

WHAT IT DOES
------------
Runs app/vjepa_3/train.py's real `main()` for a single step, on the arm's own
config and on the control's, with the same synthetic clips and masks
(tests/train/harness.py: only the data source and the process group are
substituted).  It then compares, for encoder and predictor:

  * the size of the gradient (root mean square over all elements);
  * the share of gradient elements below Adam's epsilon (1e-8), which the
    optimizer barely acts on;
  * for an arm that starts from the control's exact weights, how much of the
    gradient is NEW (the norm of the difference over the control's norm) --
    for a regulariser arm this is the regulariser's share of the push.

WHAT IT CANNOT TELL YOU
-----------------------
It is one step, in float32, on synthetic clips.  It does not see float16
overflow or underflow, behaviour after the first step, or anything that needs
real video.  A PASS means "starts in the same regime as the control", not
"will train well".
"""

import argparse
import math
import sys
import tempfile
from pathlib import Path

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tests.train.harness import deep_update, run_train  # noqa: E402

CONTROL = REPO_ROOT / "configs" / "train_3" / "ladder" / "a0-control.yaml"
ADAM_EPS = 1e-8
# An arm whose gradient is within this factor of the control's is in the same
# regime.  n1-dyt was off by 8,000-13,000x; nothing healthy was off by 10x.
RATIO_LIMIT = 30.0


def one_step(cfg_path, folder, tiny=False, batch_size=2):
    """Run a single training step of the config at `cfg_path`; return its numbers."""
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    cfg["folder"] = str(folder)
    deep_update(
        cfg,
        {
            "data": {"batch_size": batch_size, "num_workers": 0, "pin_mem": False},
            "meta": {"dtype": "float32", "sigreg_diag_freq": 0, "load_checkpoint": False},
            "optimization": {"epochs": 1, "ipe": 1},
            # Full context-loss weight from the first step.  In a real run it
            # is ramped in later; here it must be on, or half of the loss
            # would go unchecked.
            "model": {"lambda_progressive": False},
        },
    )
    if tiny:
        deep_update(
            cfg,
            {
                "data": {"crop_size": 64, "dataset_fpcs": [8]},
                "model": {
                    "model_name": "vit_tiny",
                    "pred_depth": 4,
                    "pred_embed_dim": 96,
                    "pred_num_heads": 3,
                },
            },
        )
    grads = {}

    def grab(encoder, predictor):
        for label, model in (("encoder", encoder), ("predictor", predictor)):
            grads[label] = {
                name: p.grad.detach().clone()
                for name, p in model.named_parameters()
                if p.grad is not None
            }

    out = run_train(cfg, n_clips=batch_size, on_optimizer_step=grab, clip_kind="smooth")
    return {
        "loss": float(out["latest"]["loss"]),
        "grads": grads,
        "init": {"encoder": out["init_encoder"], "predictor": out["init_predictor"]},
    }


def summarize(grads):
    flat = torch.cat([g.flatten().double() for g in grads.values()])
    return {
        "rms": float(flat.square().mean().sqrt()),
        "below_eps": float((flat.abs() < ADAM_EPS).double().mean()),
        "finite": bool(torch.isfinite(flat).all()),
    }


def new_share(arm, control, label):
    """||g_arm - g_control|| / ||g_control|| when the two start identical, else None."""
    a, c = arm["init"][label], control["init"][label]
    if a.keys() != c.keys() or any(a[k].shape != c[k].shape or not torch.equal(a[k], c[k]) for k in a):
        return None
    ga, gc = arm["grads"][label], control["grads"][label]
    if ga.keys() != gc.keys():
        return None
    num = sum(float((ga[k].double() - gc[k].double()).square().sum()) for k in gc)
    den = sum(float(gc[k].double().square().sum()) for k in gc)
    return math.sqrt(num / den) if den > 0 else None


def check(arm, control):
    """Compare one arm with the control.  Returns (rows, problems)."""
    rows, problems = [], []
    if not math.isfinite(arm["loss"]):
        problems.append(f"loss is {arm['loss']}")
    for label in ("encoder", "predictor"):
        a, c = summarize(arm["grads"][label]), summarize(control["grads"][label])
        ratio = a["rms"] / c["rms"] if c["rms"] > 0 else float("inf")
        share = new_share(arm, control, label)
        rows.append((label, a["rms"], ratio, a["below_eps"], c["below_eps"], share))
        if not a["finite"]:
            problems.append(f"{label} gradient has non-finite values")
        if not (1.0 / RATIO_LIMIT <= ratio <= RATIO_LIMIT):
            problems.append(
                f"{label} gradient is {ratio:.3g}x the control's (limit {RATIO_LIMIT:g}x either way)"
            )
        if a["below_eps"] - c["below_eps"] > 0.25:
            problems.append(
                f"{100 * a['below_eps']:.0f}% of {label} gradient elements are below Adam's "
                f"epsilon (control: {100 * c['below_eps']:.0f}%)"
            )
    return rows, problems


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("configs", nargs="+", help="arm config file(s)")
    parser.add_argument("--control", default=str(CONTROL), help="control config file")
    parser.add_argument(
        "--tiny", action="store_true", help="shrink the model (fast; for testing this script)"
    )
    args = parser.parse_args(argv)

    torch.set_num_threads(max(1, torch.get_num_threads()))
    failed = 0
    with tempfile.TemporaryDirectory() as tmp:
        control = one_step(args.control, Path(tmp) / "control", tiny=args.tiny)
        print(f"control  {Path(args.control).name}: loss {control['loss']:.4f}")
        for index, path in enumerate(args.configs):
            name = Path(path).stem
            if Path(path).resolve() == Path(args.control).resolve():
                continue
            pretrained = (yaml.safe_load(Path(path).read_text()).get("meta") or {}).get(
                "pretrained_encoder"
            )
            if pretrained and not Path(pretrained).exists():
                print(f"\n{name}: SKIP - needs the encoder file {pretrained}, which is not here")
                continue
            try:
                arm = one_step(path, Path(tmp) / f"arm{index}", tiny=args.tiny)
            except Exception as error:  # a config that cannot take one step has failed
                print(f"\n{name}: FAIL - could not run one step: {type(error).__name__}: {error}")
                failed += 1
                continue
            rows, problems = check(arm, control)
            print(f"\n{name}: loss {arm['loss']:.4f}")
            print(f"  {'':10s} {'grad rms':>10s} {'vs control':>11s} {'below eps':>10s} {'(control)':>10s} {'new share':>10s}")
            for label, rms, ratio, below, below_c, share in rows:
                share_text = "    n/a" if share is None else f"{share:10.3g}"
                print(
                    f"  {label:10s} {rms:10.3g} {ratio:10.3g}x {100 * below:9.1f}% "
                    f"{100 * below_c:9.1f}% {share_text}"
                )
            if problems:
                failed += 1
                for problem in problems:
                    print(f"  WARN  {problem}")
            else:
                print("  PASS  starts in the same regime as the control")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
