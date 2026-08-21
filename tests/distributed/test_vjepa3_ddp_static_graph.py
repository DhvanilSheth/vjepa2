# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""Does Cayley-STRING survive DistributedDataParallel(static_graph=True)?

THE HAZARD -- stated precisely
------------------------------
app/vjepa_3/train.py wraps the encoder as

    DistributedDataParallel(encoder, static_graph=True)

Under static_graph, DDP records the autograd graph on the first iteration and
reuses the reduction schedule forever after.

MEASURED on torch 2.13 (not assumed -- the two cases behave differently):

  (A) a parameter CONSISTENTLY unused on a rank, same every iteration
      -> ACCEPTED.  static_graph explicitly supports unused parameters, so
         long as the used-set is constant.  No error, no hang.
         Caveat: it is still not harmless.  A rank contributing no gradient
         still participates in the allreduce, so the averaged gradient for
         that parameter is scaled down by the fraction of ranks that skipped
         it -- a silent change in effective learning rate, not a crash.

  (B) usage VARYING ACROSS ITERATIONS on one rank
      -> RuntimeError("Expected to have finished reduction in the prior
         iteration before starting a new one. This error indicates that your
         training graph has changed...") on the offending rank, and a HANG on
         its peer, which is waiting on an allreduce that never comes.

So the real rule is not "P must be used on every step" but "the used-parameter
set must be CONSTANT ACROSS ITERATIONS on each rank."  V-JEPA's own rank split
(`img_rank_ratio` in train.py:263-289) assigns a rank to images or video for
the whole run, which is case (A) -- so a modality-gated P would degrade
silently rather than crash.  That is arguably worse.

V-JEPA 3 sidesteps both: P is computed unconditionally, from one always-live
parameter, and applied to every token of every head in every layer.

WHAT THIS ASSERTS
-----------------
positive: the real model -- P computed unconditionally, applied to every token
          of every head in every layer -- trains across a T=1 rank and a
          T=8 rank under static_graph=True.

negative: a deliberately gated variant that toggles STRING on and off across
          ITERATIONS -- case (B) above -- MUST fail.  Without this control the
          positive test proves nothing: a test that cannot fail is not
          evidence.  An earlier version of this control gated on T == 1, i.e.
          case (A), and PASSED -- which is exactly how a rubber-stamp control
          hides a real hazard.

RUN
---
    python3 -m pytest tests/distributed/test_vjepa3_ddp_static_graph.py -v
    python3 tests/distributed/test_vjepa3_ddp_static_graph.py   # direct

CPU + gloo, no GPU required.  ~15s: the negative arm short-circuits as
soon as one rank faults rather than waiting out its peer's hang.
"""

import os
import sys
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

CROP = 64
VIDEO_FRAMES = 8
N_STEPS = 4  # even, so the alternating control toggles at least twice
TIMEOUT_S = 90

# Set by the worker before each step; read by the patched apply_string.
_SKIP_STRING = [False]


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _worker(rank, world_size, gated, port, out):
    """One DDP rank.  rank 0 feeds T=1 images, rank 1 feeds T=8 video."""
    try:
        # mp spawn starts a FRESH interpreter that does not inherit sys.path,
        # so `import app...` fails without this.  Must precede the imports.
        if REPO_ROOT not in sys.path:
            sys.path.insert(0, REPO_ROOT)

        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(port)
        dist.init_process_group("gloo", rank=rank, world_size=world_size)

        import app.vjepa_3.models.utils.modules as modules_mod
        import app.vjepa_3.models.vision_transformer as vit_3

        if gated:
            # NEGATIVE CONTROL -- case (B): toggle STRING ACROSS ITERATIONS so
            # the used-parameter set is not constant.  Gating on T alone
            # (case A) is accepted by static_graph and would make this control
            # useless.  Production code is untouched; the gate lives only in
            # this patched wrapper.
            real_apply = modules_mod.apply_string

            def gated_apply(x, basis):
                return x if _SKIP_STRING[0] else real_apply(x, basis)

            modules_mod.apply_string = gated_apply

        torch.manual_seed(0)
        model = vit_3.vit_tiny(
            img_size=(CROP, CROP),
            num_frames=VIDEO_FRAMES,
            use_rope=True,
            img_temporal_dim_size=1,
            use_string=True,
            string_init_std=0.1,
        )
        ddp = DistributedDataParallel(model, static_graph=True)
        opt = torch.optim.SGD(ddp.parameters(), lr=1e-3)

        # rank 0 is the "image rank", rank 1 the "video rank" -- the exact
        # split train.py creates via img_rank_ratio.
        n_frames = 1 if rank == 0 else VIDEO_FRAMES

        for step in range(N_STEPS):
            # Constant used-set in the positive arm; alternating in the
            # negative arm, which is the pattern static_graph actually forbids.
            _SKIP_STRING[0] = gated and (rank == 0) and (step % 2 == 1)
            opt.zero_grad()
            x = torch.randn(1, 3, n_frames, CROP, CROP)
            ddp(x)[-1].sum().backward()
            opt.step()

        grad = model.string.weight.grad
        out.put(
            (
                rank,
                "ok",
                None if grad is None else float(grad.abs().max()),
            )
        )
    except Exception as exc:  # noqa: BLE001 -- the failure mode IS the result
        out.put((rank, f"{type(exc).__name__}: {exc}"[:400], None))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _run(gated, port):
    """Spawn 2 ranks; return (all_ok, [messages])."""
    ctx = mp.get_context("spawn")
    out = ctx.Queue()
    procs = [
        ctx.Process(target=_worker, args=(rank, 2, gated, port, out))
        for rank in range(2)
    ]
    for p in procs:
        p.start()

    results = []
    for _ in range(2):
        try:
            results.append(out.get(timeout=TIMEOUT_S))
        except Exception:
            results.append((-1, "TIMEOUT (DDP deadlock)", None))
            break
        # Short-circuit: when one rank reports a fault its peer is typically
        # blocked on an allreduce that will never arrive, so waiting out the
        # full timeout only makes the suite slow.  One fault is the verdict.
        if results[-1][1] != "ok":
            results.append((-1, "peer abandoned (expected: hung on allreduce)", None))
            break
    for p in procs:
        p.join(timeout=10)
        if p.is_alive():
            p.terminate()
            p.join(timeout=5)

    messages = [f"rank {r}: {status}" for r, status, _ in results]
    all_ok = len(results) == 2 and all(status == "ok" for _, status, _ in results)
    return all_ok, messages, results


class TestDDPStaticGraph(unittest.TestCase):
    def test_positive_string_survives_static_graph(self):
        """The real model: P used unconditionally, so all ranks agree."""
        ok, messages, results = _run(gated=False, port=29551)
        self.assertTrue(ok, "STRING failed under static_graph:\n" + "\n".join(messages))
        for rank, _, grad_max in results:
            self.assertIsNotNone(grad_max, f"rank {rank}: no gradient on string.weight")
            self.assertGreater(
                grad_max, 0.0, f"rank {rank}: zero gradient -- P was not used"
            )

    def test_negative_control_gated_string_fails(self):
        """A gated P MUST break. If this passes, the test above proves nothing."""
        ok, messages, _ = _run(gated=True, port=29552)
        joined = "\n".join(messages)
        self.assertFalse(
            ok,
            "NEGATIVE CONTROL DID NOT FAIL -- gating STRING on T>1 was expected "
            "to break DDP(static_graph=True), so the positive test is not "
            "actually exercising the hazard:\n" + joined,
        )
        # It must fail for the RIGHT reason.  An ImportError or a typo would
        # also "fail" and would silently turn this control into a rubber stamp.
        self.assertFalse(
            any(bad in joined for bad in ("ModuleNotFoundError", "ImportError", "NameError")),
            "negative control failed for the WRONG reason (not a DDP fault):\n" + joined,
        )
        self.assertTrue(
            any(
                marker in joined
                for marker in ("RuntimeError", "TIMEOUT", "mark", "reduc", "unused", "static")
            ),
            "negative control failed, but not with a recognisable DDP fault:\n" + joined,
        )


if __name__ == "__main__":
    for gated in (False, True):
        ok, messages, _ = _run(gated=gated, port=29551 + int(gated))
        label = "negative control (gated)" if gated else "positive (real model)"
        print(f"\n=== {label}: {'PASSED' if ok else 'FAILED'} ===")
        for m in messages:
            print("   ", m)
    sys.exit(0)
