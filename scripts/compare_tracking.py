# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Compare tracking-probe results of ladder arms against a control, with error bars.

    python3 scripts/compare_tracking.py --runs /runs/pretrain_3/ladder \\
        --control a0-control [--control a0-control-s2] [--mode frame] [arm ...]

Reads the JSON files evals/video_tracking_frozen/eval.py writes
(<run>/video_tracking_frozen/davis-tracking/davis_val_<mode>_latest.json).
With no arm names, every run folder under --runs that has a result is used.

TWO SOURCES OF NOISE, AND WHAT THIS TOOL SAYS ABOUT EACH
---------------------------------------------------------
1. Test-set noise: DAVIS val is 30 videos, and one video can swing a run's
   score by more than half a point.  For each arm the per-object differences
   to the control are resampled BY VIDEO (objects in one video are not
   independent), 5,000 times; the 95% range is printed.
2. Training noise: two runs of the SAME config differ.  Give a second control
   (--control twice) and the gap between the controls is printed as the
   yardstick; an arm is only marked "clear" when its 95% range excludes 0 AND
   its difference is larger than that gap.  With more than two controls the
   largest pairwise gap is used.

With six arms and a 95% range, one arm landing just outside 0 by chance is not
unusual.  "clear" means "worth a second seed", not "proven".
"""

import argparse
import itertools
import json
import os
import sys

import numpy as np

TAG = "davis-tracking"


def result_path(runs_root, arm, mode, split="val", stem="latest"):
    return os.path.join(
        runs_root, arm, "video_tracking_frozen", TAG, f"davis_{split}_{mode}_{stem}.json"
    )


def load_scores(path):
    """{(sequence, object): J&F in points} from one probe result file."""
    with open(path, "r") as f:
        report = json.load(f)
    return {
        (row["sequence"], row["object"]): 50.0 * (row["J"] + row["F"])
        for row in report["per_object"]
    }


def paired_bootstrap(arm, control, n_boot=5000, seed=0):
    """Paired difference arm - control over shared objects, resampled by video.

    Returns {"diff", "low", "high", "better", "worse", "objects", "videos"}.
    """
    if arm.keys() != control.keys():
        raise ValueError(
            f"the two results cover different objects ({len(arm)} vs {len(control)}); "
            "they must come from the same DAVIS split"
        )
    sequences = sorted({key[0] for key in control})
    by_video = {
        seq: np.array([arm[k] - control[k] for k in sorted(control) if k[0] == seq])
        for seq in sequences
    }
    diffs = np.concatenate([by_video[seq] for seq in sequences])
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(sequences), size=(n_boot, len(sequences)))
    boot = np.array(
        [np.concatenate([by_video[sequences[i]] for i in draw]).mean() for draw in draws]
    )
    low, high = np.percentile(boot, [2.5, 97.5])
    return {
        "diff": float(diffs.mean()),
        "low": float(low),
        "high": float(high),
        "better": int((diffs > 0).sum()),
        "worse": int((diffs < 0).sum()),
        "objects": int(diffs.size),
        "videos": len(sequences),
    }


def seed_gap(controls):
    """Largest |mean difference| between any two control runs, or None."""
    gaps = [
        abs(float(np.mean([a[k] - b[k] for k in a])))
        for a, b in itertools.combinations(controls, 2)
    ]
    return max(gaps) if gaps else None


def verdict(stats, gap):
    excludes_zero = stats["low"] > 0 or stats["high"] < 0
    if not excludes_zero:
        return "not separable"
    if gap is None:
        return "outside test-set noise (training noise unknown)"
    if abs(stats["diff"]) <= gap:
        return "within training noise"
    return "clear: better" if stats["diff"] > 0 else "clear: worse"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--runs", required=True, help="folder holding one run folder per arm")
    parser.add_argument(
        "--control", action="append", required=True, help="control run name (repeat for seeds)"
    )
    parser.add_argument("--mode", default="frame", choices=["frame", "clip"])
    parser.add_argument("arms", nargs="*", help="arms to compare (default: all with a result)")
    args = parser.parse_args(argv)

    controls = [load_scores(result_path(args.runs, name, args.mode)) for name in args.control]
    reference = controls[0]
    gap = seed_gap(controls)

    arms = args.arms or sorted(
        name
        for name in os.listdir(args.runs)
        if name not in args.control and os.path.exists(result_path(args.runs, name, args.mode))
    )

    print(f"DAVIS val, {args.mode} mode, compared with {args.control[0]}")
    for name, scores in zip(args.control, controls):
        print(f"  control {name:24s} J&F {np.mean(list(scores.values())):5.1f}")
    if gap is None:
        print("  training noise: unknown (give a second --control to measure it)")
    else:
        print(f"  training noise: largest gap between controls = {gap:.1f} points")
    print()
    print(f"{'arm':24s} {'J&F':>5s} {'diff':>6s}  {'95% range':>15s}  {'better/worse':>12s}  verdict")
    for name in arms:
        scores = load_scores(result_path(args.runs, name, args.mode))
        stats = paired_bootstrap(scores, reference)
        print(
            f"{name:24s} {np.mean(list(scores.values())):5.1f} {stats['diff']:+6.1f}  "
            f"[{stats['low']:+5.1f}, {stats['high']:+5.1f}]  "
            f"{stats['better']:>5d}/{stats['worse']:<6d}  {verdict(stats, gap)}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
