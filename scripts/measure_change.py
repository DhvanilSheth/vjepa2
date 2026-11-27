# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Measure, on real video, whether "has this patch changed?" can be decided
from pixels.  CPU only; no model, no GPU.

    python3 scripts/measure_change.py --davis /data/DAVIS [--tol 1] [--out change.json]

WHY THIS EXISTS
---------------
The masking rule under design (V3_STRING/PROGRESS.md, section 8.1) never asks
the model for a patch that looks the same as a patch it was shown at the same
place: such a target can be copied without understanding anything.  That rule
needs a number d(patch at time a, same patch at time b) and a threshold below
which the patch counts as UNCHANGED.  This script measures d on DAVIS, where
each frame has a hand-drawn mask of the moving objects, to answer:

1. Is there a threshold at all?  (Do unchanged and changed patches form two
   groups, or one smear?)
2. How much of a video is unchanged at a gap of 1/24 s, 1/4 s, 1 s?
3. Does "changed" mark the patches the annotators call objects?

THE SIGNAL
----------
`patch_change` compares two grayscale frames patch by patch, after three
steps.  Each removes one way a patch can differ in pixels while showing the
same thing; all three were needed on synthetic scenes with a known answer
(tests/scripts/test_measure_change.py):

  low-pass      average `cell` x `cell` pixels: sensor and compression noise.
  brightness    subtract each patch's mean and fit one gain per patch:
                exposure changes.
  small shift   take the smallest difference over shifts of up to `tol`
                pixels: camera shake.  `tol` is also the smallest motion that
                counts as change.

A patch in the middle of a moving, evenly coloured object reads as unchanged.
That is intended: copying is correct there.
"""

import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evals.video_tracking_frozen.davis import (  # noqa: E402
    read_annotation,
    read_sequences,
    sequence_files,
)

SIZE = 256  # frames are resized (short side) and centre-cropped to SIZE x SIZE
PATCH = 16
DAVIS_FPS = 24
THRESHOLDS = (0.004, 0.006, 0.008, 0.012, 0.016, 0.024, 0.032)


def _pool(frame, cell):
    """[H, W] -> [H/cell, W/cell], the mean of each cell."""
    h, w = frame.shape
    return frame.reshape(h // cell, cell, w // cell, cell).mean((1, 3))


def _per_patch(pooled, cells):
    """[G*cells, G*cells] -> [G, G, cells, cells], each patch's mean removed."""
    g = pooled.shape[0] // cells
    patches = pooled.reshape(g, cells, g, cells).transpose(0, 2, 1, 3)
    return patches - patches.mean((2, 3), keepdims=True)


def _shifted(frame, dy, dx):
    """`frame` moved by (dy, dx) pixels; the uncovered border repeats the edge."""
    pad = max(abs(dy), abs(dx))
    if pad == 0:
        return frame
    padded = np.pad(frame, pad, mode="edge")
    h, w = frame.shape
    return padded[pad - dy : pad - dy + h, pad - dx : pad - dx + w]


def patch_change(a, b, patch=PATCH, cell=4, tol=1, gain=(0.8, 1.25)):
    """How much each patch differs between two frames.

    a, b: [H, W] grayscale in [0, 1], H and W multiples of `patch`.
    Returns [H/patch, W/patch]; 0 means identical.  Units are grey levels on
    the 0-1 scale, so 0.01 is about 2.5 of 255.
    """
    cells = patch // cell
    pa = _per_patch(_pool(a, cell), cells)
    energy = np.maximum((pa * pa).sum((2, 3), keepdims=True), 1e-8)
    best = None
    for dy in range(-tol, tol + 1):
        for dx in range(-tol, tol + 1):
            pb = _per_patch(_pool(_shifted(b, dy, dx), cell), cells)
            g = np.clip((pa * pb).sum((2, 3), keepdims=True) / energy, *gain)
            d = np.abs(pb - g * pa).mean((2, 3))
            best = d if best is None else np.minimum(best, d)
    return best


def _square(image, resample):
    """Resize the short side to SIZE and centre-crop to SIZE x SIZE."""
    scale = SIZE / min(image.size)
    w, h = (max(SIZE, round(s * scale)) for s in image.size)
    image = image.resize((w, h), resample)
    left, top = (w - SIZE) // 2, (h - SIZE) // 2
    return image.crop((left, top, left + SIZE, top + SIZE))


def load_gray(path):
    return np.asarray(_square(Image.open(path).convert("L"), Image.BICUBIC), np.float32) / 255.0


def load_object_share(path):
    """[G, G] share of each patch covered by any annotated object."""
    ids = read_annotation(path)
    covered = np.asarray(_square(Image.fromarray((ids > 0).astype(np.uint8)), Image.NEAREST))
    g = SIZE // PATCH
    return covered.reshape(g, PATCH, g, PATCH).mean((1, 3))


def measure_sequence(frame_paths, mask_paths, gaps, tol, stride=4):
    """{gap: {"background": [d ...], "object": [d ...]}} for one sequence.

    A patch is "object" if objects cover at least a quarter of it in either
    frame of the pair, "background" if they cover none of it in both; patches
    in between are left out.
    """
    frames = [load_gray(p) for p in frame_paths]
    shares = [load_object_share(p) for p in mask_paths]
    out = {}
    for gap in gaps:
        background, objects = [], []
        for t in range(0, len(frames) - gap, stride):
            d = patch_change(frames[t], frames[t + gap], tol=tol)
            share = np.maximum(shares[t], shares[t + gap])
            background.append(d[share == 0])
            objects.append(d[share >= 0.25])
        if background:
            out[gap] = {
                "background": np.concatenate(background),
                "object": np.concatenate(objects),
            }
    return out


def _share_below(values, threshold):
    return float((values < threshold).mean()) if len(values) else float("nan")


def report(per_sequence, gaps, print_fn=print):
    """Print the three tables; return them as a dict for --out."""
    pooled = {
        gap: {
            kind: np.concatenate([s[gap][kind] for s in per_sequence.values() if gap in s])
            for kind in ("background", "object")
        }
        for gap in gaps
    }
    result = {"thresholds": list(THRESHOLDS), "gaps": {}, "sequences": {}}

    print_fn("1. Share of patches that read as UNCHANGED, by threshold and time gap")
    print_fn("   (background should be high where the camera is still; object should be low)")
    header = "   gap        kind        " + "".join(f"{t:>8.3f}" for t in THRESHOLDS)
    print_fn(header)
    for gap in gaps:
        entry = {}
        for kind in ("background", "object"):
            shares = [_share_below(pooled[gap][kind], t) for t in THRESHOLDS]
            entry[kind] = shares
            print_fn(
                f"   {gap / DAVIS_FPS:5.3f} s    {kind:10s}  "
                + "".join(f"{100 * s:7.0f}%" for s in shares)
            )
        result["gaps"][gap] = entry

    mid = gaps[len(gaps) // 2]
    print_fn("")
    print_fn(f"2. Is there a gap between two groups?  All patches at {mid / DAVIS_FPS:.3f} s,")
    print_fn("   counted by size of change (each row is a factor of 1.6):")
    values = np.concatenate([pooled[mid]["background"], pooled[mid]["object"]])
    edges = 0.001 * 1.6 ** np.arange(0, 13)
    counts, _ = np.histogram(np.clip(values, edges[0], edges[-1] - 1e-9), edges)
    for low, high, count in zip(edges[:-1], edges[1:], counts):
        bar = "#" * int(round(60 * count / max(1, counts.max())))
        print_fn(f"   {low:.4f}-{high:.4f}  {100 * count / len(values):5.1f}%  {bar}")
    result["histogram"] = {"edges": edges.tolist(), "counts": counts.tolist(), "gap": mid}

    threshold = THRESHOLDS[len(THRESHOLDS) // 2]
    print_fn("")
    print_fn(f"3. Per video at {mid / DAVIS_FPS:.3f} s, threshold {threshold}: share CHANGED")
    print_fn("   (background near 100% means the camera moves)")
    print_fn(f"   {'video':22s} {'background':>10s} {'object':>8s}")
    for name, by_gap in sorted(per_sequence.items()):
        if mid not in by_gap:
            continue
        row = {
            kind: 1.0 - _share_below(by_gap[mid][kind], threshold)
            for kind in ("background", "object")
        }
        result["sequences"][name] = row
        print_fn(f"   {name:22s} {100 * row['background']:9.0f}% {100 * row['object']:7.0f}%")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--davis", required=True, help="DAVIS root folder")
    parser.add_argument("--split", default="val")
    parser.add_argument(
        "--gaps", type=int, nargs="+", default=[1, 3, 6, 12, 24], help="in frames (24 fps)"
    )
    parser.add_argument("--tol", type=int, default=1, help="shift tolerance in pixels")
    parser.add_argument("--out", help="also write the tables to this JSON file")
    args = parser.parse_args(argv)

    per_sequence = {}
    for name in read_sequences(args.davis, args.split):
        per_sequence[name] = measure_sequence(
            *sequence_files(args.davis, name), gaps=args.gaps, tol=args.tol
        )
    print(f"DAVIS {args.split}: {len(per_sequence)} videos, shift tolerance {args.tol} px")
    result = report(per_sequence, args.gaps)
    result["tol"] = args.tol
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
