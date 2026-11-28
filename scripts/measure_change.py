# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Measure, on real video, whether "has this patch changed?" can be decided
from pixels.  CPU only; no model, no GPU.

    python3 scripts/measure_change.py --davis /data/DAVIS [--tol 1] [--out change.json]
    python3 scripts/measure_change.py --videos train_paths.csv [--n-videos 200]

--davis has a hand-drawn mask of the moving objects in every frame, so patches
can be split into "object" and "background".  --videos reads a sample of the
training videos (the list `data.datasets` points at); it has no masks, so all
patches are counted together.

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
The signal lives in app/vjepa_3/change.py, because training uses the same
functions.  `patch_change` compares two grayscale frames patch by patch, after
three steps.  Each removes one way a patch can differ in pixels while showing the
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

THE SHARED MOTION
-----------------
When the camera moves, every patch changes.  Telling camera motion from object
motion exactly is not always possible from pixels (an object that fills the
frame, an object moving along the camera's path, a camera moving through
depth), and the masking rule does not need it.  It needs to know which
patches are TRIVIAL to predict.  So the question asked here is: is this patch
explained by the one motion most of the frame shares?

`shared_shift` estimates that motion as one shift for the whole frame, and the
change is measured again after undoing it.  What still reads as changed is not
explained by the shared motion: an object moving on its own, or depth, zoom
and rotation, which one shift does not describe.

`unexplained_weight` turns the remaining change into a weight between 0 and 1.
The measurements show no natural threshold, so the rule is soft: a patch is
asked for in proportion to how much the shared motion fails to explain it.
"""

import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.vjepa_3.change import (  # noqa: E402,F401  (re-exported for the tests)
    _shifted,
    covered_patches,
    patch_change,
    shared_shift,
    unexplained_weight,
)
from evals.video_tracking_frozen.davis import (  # noqa: E402
    read_annotation,
    read_sequences,
    sequence_files,
)

SIZE = 256  # frames are resized (short side) and centre-cropped to SIZE x SIZE
PATCH = 16
DAVIS_FPS = 24
THRESHOLDS = (0.004, 0.006, 0.008, 0.012, 0.016, 0.024, 0.032)


WEIGHT_FULL = (0.016, 0.032, 0.064)  # candidate values of `full`

# How much of a video the shared motion explains, from the share of patches
# that read as changed before and after undoing it.
REGIMES = (
    "still: little changes at all",
    "explained: the shared motion accounts for it",
    "partly explained",
    "not explained: depth, zoom, or much moves on its own",
)


def regime(changed, changed_after):
    if changed < 0.3:
        return REGIMES[0]
    if changed_after < 0.3:
        return REGIMES[1]
    return REGIMES[2] if changed_after < 0.7 else REGIMES[3]


def _square(image, resample):
    """Resize the short side to SIZE and centre-crop to SIZE x SIZE."""
    scale = SIZE / min(image.size)
    w, h = (max(SIZE, round(s * scale)) for s in image.size)
    image = image.resize((w, h), resample)
    left, top = (w - SIZE) // 2, (h - SIZE) // 2
    return image.crop((left, top, left + SIZE, top + SIZE))


def load_gray(path):
    return np.asarray(_square(Image.open(path).convert("L"), Image.BICUBIC), np.float32) / 255.0


def load_object_mask(path):
    """[SIZE, SIZE] float, 1 where any annotated object is."""
    ids = read_annotation(path)
    covered = _square(Image.fromarray((ids > 0).astype(np.uint8)), Image.NEAREST)
    return np.asarray(covered, np.float32)


def object_share(mask):
    """[G, G] share of each patch covered by an object."""
    g = SIZE // PATCH
    return mask.reshape(g, PATCH, g, PATCH).mean((1, 3))


DAVIS_KINDS = ("background", "object")
VIDEO_KINDS = ("all",)


def measure_pair(a, b, tol, mask_a=None, mask_b=None):
    """Change of every patch between two frames, as measured and after undoing
    the shared motion.

    With masks: {"background", "object", "background_shared", "object_shared"}.
    A patch is "object" if objects cover at least a quarter of it in either
    frame, "background" if they cover none of it in both; patches in between
    are left out.  Without masks: {"all", "all_shared"}.  Always "shift", the
    pixels of the shared shift.
    """
    dy, dx = shared_shift(a, b)
    raw = patch_change(a, b, tol=tol)
    undone = patch_change(a, _shifted(b, dy, dx), tol=tol)
    valid = covered_patches(dy, dx)
    out = {"shift": float(np.hypot(dy, dx))}
    if mask_a is None:
        out["all"], out["all_shared"] = raw.ravel(), undone[valid]
        return out
    share = np.maximum(object_share(mask_a), object_share(mask_b))
    out["background"], out["object"] = raw[share == 0], raw[share >= 0.25]
    share = np.maximum(object_share(mask_a), object_share(_shifted(mask_b, dy, dx)))
    out["background_shared"] = undone[valid & (share == 0)]
    out["object_shared"] = undone[valid & (share >= 0.25)]
    return out


def _gather(pairs):
    """[measure_pair result, ...] -> one dict of concatenated arrays."""
    return {
        key: np.array([p[key] for p in pairs])
        if key == "shift"
        else np.concatenate([p[key] for p in pairs])
        for key in pairs[0]
    }


def measure_sequence(frame_paths, mask_paths, gaps, tol, stride=4):
    """{gap in frames: gathered measure_pair results} for one DAVIS sequence."""
    frames = [load_gray(p) for p in frame_paths]
    masks = [load_object_mask(p) for p in mask_paths]
    out = {}
    for gap in gaps:
        pairs = [
            measure_pair(frames[t], frames[t + gap], tol, masks[t], masks[t + gap])
            for t in range(0, len(frames) - gap, stride)
        ]
        if pairs:
            out[gap] = _gather(pairs)
    return out


def _open_with_decord(path):
    """(number of frames, frames per second, function: indices -> [n, H, W, 3] uint8)."""
    from decord import VideoReader, cpu  # the reader the training loader uses

    reader = VideoReader(path, num_threads=1, ctx=cpu(0))
    return len(reader), float(reader.get_avg_fps()), lambda idx: reader.get_batch(idx).asnumpy()


def measure_video(path, gaps_s, tol, pairs, rng, open_video=None):
    """{gap in seconds: gathered measure_pair results} for one video file.

    For each gap, `pairs` frame pairs are taken at random start times.  A gap
    shorter than one frame is measured at one frame; a gap longer than the
    video is left out.
    """
    n_frames, fps, read = (open_video or _open_with_decord)(path)
    wanted = {}
    for gap in gaps_s:
        step = max(1, int(round(gap * fps)))
        if step < n_frames:
            starts = rng.integers(0, n_frames - step, size=pairs)
            wanted[gap] = [(int(t), int(t) + step) for t in starts]
    indices = sorted({i for pair_list in wanted.values() for pair in pair_list for i in pair})
    if not indices:
        return {}
    frames = {
        i: np.asarray(_square(Image.fromarray(f).convert("L"), Image.BICUBIC), np.float32) / 255.0
        for i, f in zip(indices, read(indices))
    }
    return {
        gap: _gather([measure_pair(frames[i], frames[j], tol) for i, j in pair_list])
        for gap, pair_list in wanted.items()
    }


def read_video_list(csv_path):
    """Paths from a training list: one "<path> <label>" per line."""
    with open(csv_path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]
    return [line.rsplit(" ", 1)[0] if " " in line else line for line in lines]


def _share_below(values, threshold):
    return float((values < threshold).mean()) if len(values) else float("nan")


def _histogram(values, print_fn):
    edges = 0.001 * 1.6 ** np.arange(0, 13)
    counts, _ = np.histogram(np.clip(values, edges[0], edges[-1] - 1e-9), edges)
    for low, high, count in zip(edges[:-1], edges[1:], counts):
        bar = "#" * int(round(60 * count / max(1, counts.max())))
        print_fn(f"   {low:.4f}-{high:.4f}  {100 * count / len(values):5.1f}%  {bar}")
    return {"edges": edges.tolist(), "counts": counts.tolist()}


def report(per_video, gaps, kinds, seconds, list_videos=True, print_fn=print):
    """Print the tables; return them as a dict for --out.

    per_video: {name: {gap: gathered measure_pair results}}.
    seconds:   function turning a gap key into seconds.
    """
    columns = [kind + suffix for kind in kinds for suffix in ("", "_shared")]
    pooled = {
        gap: {
            col: np.concatenate([v[gap][col] for v in per_video.values() if gap in v])
            for col in columns
        }
        for gap in gaps
    }
    result = {"thresholds": list(THRESHOLDS), "gaps": {}, "videos": {}, "after_shared": {}}
    threshold = THRESHOLDS[len(THRESHOLDS) // 2]
    mid = gaps[len(gaps) // 2]
    head = "".join(f" {col.replace('_shared', ' -> after'):>19s}" for col in columns)

    print_fn("1. Share of patches that read as UNCHANGED, by threshold and time gap")
    print_fn("   gap        kind        " + "".join(f"{t:>8.3f}" for t in THRESHOLDS))
    for gap in gaps:
        result["gaps"][gap] = {}
        for kind in kinds:
            shares = [_share_below(pooled[gap][kind], t) for t in THRESHOLDS]
            result["gaps"][gap][kind] = shares
            print_fn(
                f"   {seconds(gap):5.3f} s    {kind:10s}  "
                + "".join(f"{100 * s:7.0f}%" for s in shares)
            )

    print_fn("")
    print_fn(f"2. Is there a gap between two groups?  All patches at {seconds(mid):.3f} s,")
    print_fn("   counted by size of change (each row is a factor of 1.6).  As measured:")
    result["histogram"] = _histogram(
        np.concatenate([pooled[mid][kind] for kind in kinds]), print_fn
    )
    print_fn("   After undoing the shared motion:")
    result["histogram_shared"] = _histogram(
        np.concatenate([pooled[mid][kind + "_shared"] for kind in kinds]), print_fn
    )

    rows = {}
    for name, by_gap in sorted(per_video.items()):
        if mid in by_gap:
            rows[name] = {
                col: 1.0 - _share_below(by_gap[mid][col], threshold) for col in columns
            }
            rows[name]["shift"] = float(np.median(by_gap[mid]["shift"]))
    result["videos"] = rows
    print_fn("")
    print_fn(f"3. Per video at {seconds(mid):.3f} s, threshold {threshold}: share CHANGED,")
    print_fn("   as measured and after undoing the shared motion")
    print_fn(f"   {'video':20s} {'shift px':>9s}{head}")
    if list_videos:
        table = [(name, row) for name, row in rows.items()]
    else:  # too many to list: show the spread over videos instead
        table = [
            (
                f"{q}% of videos below",
                {key: float(np.nanpercentile([r[key] for r in rows.values()], q)) for key in ["shift"] + columns},
            )
            for q in (10, 25, 50, 75, 90)
        ]
    for name, row in table:
        print_fn(
            f"   {name:20s} {row['shift']:9.0f}"
            + "".join(f" {100 * row[col]:18.0f}%" for col in columns)
        )

    print_fn("")
    print_fn(f"4. All videos, threshold {threshold}: share CHANGED by time gap")
    print_fn(f"   {'gap':>8s}{head}")
    for gap in gaps:
        row = {col: 1.0 - _share_below(pooled[gap][col], threshold) for col in columns}
        result["after_shared"][gap] = row
        print_fn(
            f"   {seconds(gap):6.3f} s" + "".join(f" {100 * row[col]:18.0f}%" for col in columns)
        )

    print_fn("")
    print_fn(f"5. What kind of video is it?  At {seconds(mid):.3f} s, threshold {threshold}")
    counts = dict.fromkeys(REGIMES, 0)
    for by_gap in per_video.values():
        if mid in by_gap:
            before = np.concatenate([by_gap[mid][kind] for kind in kinds])
            after = np.concatenate([by_gap[mid][kind + "_shared"] for kind in kinds])
            counts[regime(1 - _share_below(before, threshold), 1 - _share_below(after, threshold))] += 1
    for name, count in counts.items():
        print_fn(f"   {count:5d} videos ({100 * count / max(1, sum(counts.values())):3.0f}%)  {name}")
    result["regimes"] = counts

    if "object" in kinds:
        print_fn("")
        print_fn(f"6. Does the weight land on objects?  At {seconds(mid):.3f} s")
        n_obj, n_bg = len(pooled[mid]["object"]), len(pooled[mid]["background"])
        print_fn(f"   objects are {100 * n_obj / (n_obj + n_bg):.0f}% of the patches; their share of the weight:")
        print_fn(f"   {'full weight at':>16s} {'as measured':>13s} {'after undoing':>15s}")
        result["weight_on_objects"] = {"by_count": n_obj / (n_obj + n_bg)}
        for full in WEIGHT_FULL:
            row = {}
            for suffix in ("", "_shared"):
                on_obj = unexplained_weight(pooled[mid]["object" + suffix], full).sum()
                on_bg = unexplained_weight(pooled[mid]["background" + suffix], full).sum()
                row[suffix or "raw"] = float(on_obj / (on_obj + on_bg))
            result["weight_on_objects"][full] = row
            print_fn(f"   {full:16.3f} {100 * row['raw']:12.0f}% {100 * row['_shared']:14.0f}%")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--davis", help="DAVIS root folder")
    source.add_argument("--videos", help='training list: one "<path> <label>" per line')
    parser.add_argument("--split", default="val", help="--davis: which split")
    parser.add_argument(
        "--gaps", type=int, nargs="+", default=[1, 3, 6, 12, 24], help="--davis: in frames"
    )
    parser.add_argument(
        "--gaps-s",
        type=float,
        nargs="+",
        default=[0.04, 0.125, 0.25, 0.5, 1.0],
        help="--videos: in seconds",
    )
    parser.add_argument("--n-videos", type=int, default=200, help="--videos: how many to sample")
    parser.add_argument("--pairs", type=int, default=4, help="--videos: frame pairs per gap")
    parser.add_argument("--seed", type=int, default=0, help="--videos: which sample")
    parser.add_argument("--tol", type=int, default=1, help="shift tolerance in pixels")
    parser.add_argument("--out", help="also write the tables to this JSON file")
    args = parser.parse_args(argv)

    per_video = {}
    if args.davis:
        for name in read_sequences(args.davis, args.split):
            per_video[name] = measure_sequence(
                *sequence_files(args.davis, name), gaps=args.gaps, tol=args.tol
            )
        print(f"DAVIS {args.split}: {len(per_video)} videos, shift tolerance {args.tol} px")
        result = report(per_video, args.gaps, DAVIS_KINDS, lambda gap: gap / DAVIS_FPS)
    else:
        rng = np.random.default_rng(args.seed)
        paths = read_video_list(args.videos)
        chosen = rng.choice(len(paths), size=min(args.n_videos, len(paths)), replace=False)
        unreadable = 0
        for index in sorted(chosen):
            try:
                found = measure_video(paths[index], args.gaps_s, args.tol, args.pairs, rng)
            except Exception as error:  # one broken file must not stop the measurement
                unreadable += 1
                print(f"skipped {paths[index]}: {error}", file=sys.stderr)
                continue
            if found:
                per_video[paths[index]] = found
        print(
            f"{len(per_video)} of {len(paths)} training videos ({unreadable} unreadable), "
            f"shift tolerance {args.tol} px"
        )
        result = report(per_video, args.gaps_s, VIDEO_KINDS, lambda gap: gap, list_videos=False)
    result["tol"] = args.tol
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
