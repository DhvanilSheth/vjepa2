# V-JEPA 3: progress record

Last updated: 2026-11-26.

This file tracks what has been done, what it showed, and what comes next.
Details live elsewhere: [README.md](README.md) explains the code and every
flag, and [LADDER_AUDIT.md](LADDER_AUDIT.md) lists every audit finding by ID.

## 1. Goal

Find out which changes to V-JEPA 2.1 measurably improve how well it
understands motion and tracks objects, with evidence we can trust, on one GPU.

What this can and cannot show:

- It **can** show "change X improves tracking over a matched, unchanged model".
- It **cannot** show "better than Meta's published model": that model trained
  for 135,000 iterations on far more data and hardware.

The plan has four steps:

1. Screen changes one at a time against a control (the "ladder").
2. Score every run on object tracking.
3. Rank the changes against the measured noise and combine the winners.
4. Confirm the combination with one long run.

## 2. Status

| Step | State |
|---|---|
| Code for the first eight arms, behind flags, tested | done (commits `139fc76` to `1b8271f`) |
| First ladder: eight 5-hour runs on the server | 7 finished; `n2-derf` stopped at epoch 3 on evidence |
| Tracking scores, single-frame mode | done for the 7 finished runs |
| Audit of theory against implementation | done; no implementation bug in any arm |
| Second control run (training noise) | **running** (`a0-control-s2`, seed 1239) |
| Tracking scores, 16-frame mode | **not started**; never run at full size on the GPU |
| Ranking against noise | waiting on the two rows above |
| Second round built: regularisers, released-encoder start, tools, fixes | done, pushed (`cbd11a0`) |
| Time-focused arms (future prediction, 12 fps, whole-frame masks, whole-clip use), from scratch and on the released encoder | done, pushed; 259 tests pass |
| Second round run | not started |
| Combination and long confirming run | not started |

## 3. Timeline

| Date | What happened |
|---|---|
| 2026-07 | Papers collected; analysis of V-JEPA 2.1 (`V2_1_ANALYSIS/`). |
| 2026-11-16 | Fork `app/vjepa_3` with RoPE frequency control, Cayley-STRING, FFN and norm flags, the isotropy reading, a clustering probe, and the ladder generator. Every flag defaults to 2.1 behaviour. |
| 2026-11-24 | Screening recipe fixed for one Quadro RTX 8000 (float16, batch 16, 9,000 iterations, about 5 h a run). Tracking probe on DAVIS 2017 added. |
| 2026-11-24 to 26 | First ladder on the server. The machine rebooted on 2026-11-25 17:01 during `p3-string`; that run was resumed from its epoch-14 checkpoint. |
| 2026-11-25 | First tracking scores (`a0`, `p1`, `p2`). |
| 2026-11-26 | Six-reviewer audit of all eight arms (`LADDER_AUDIT.md`). Server checks 1–4 run. `n1-dyt` found collapsed; `n2-derf` stopped. Remaining arms scored. Second control started. Second round built (section 7). |

## 4. Results of the first ladder

DAVIS 2017 val, 61 objects in 30 videos, single-frame mode, epoch 30. The
range is the 95% range of the difference to the control when the 30 videos
are resampled. It covers test-set noise only.

| Run | What changed | J&F | vs control | 95% range | Objects better / worse | Reading |
|---|---|---|---|---|---|---|
| `f2-relu2` | squared-ReLU feed-forward | 36.2 | +1.4 | +0.1 to +2.9 | 32 / 29 | weak positive; a few objects carry it |
| `p2-theta-temporal` | RoPE base 100 on time only | 35.2 | +0.3 | −0.6 to +1.2 | 31 / 30 | not tested yet: this mode cannot see time |
| **`a0-control`** | nothing | **34.9** | – | – | – | – |
| `p1-theta100` | RoPE base 100 on all axes | 34.2 | −0.6 | −1.6 to +0.5 | 25 / 36 | no evidence of a gain |
| `f1-swiglu` | SwiGLU feed-forward | 33.9 | −1.0 | −2.1 to +0.3 | 19 / 42 | no evidence of a gain; different start from the control |
| `p3-string` | Cayley-STRING learned basis | 33.5 | −1.4 | −2.6 to −0.4 | 24 / 37 | worse; and adds no capacity (audit S-01) |
| untrained encoder | – | 28.1 | −6.8 | – | – | the floor of this test |
| `n1-dyt` | Dynamic Tanh for LayerNorm | 18.7 | −16.2 | −20.3 to −12.4 | 3 / 58 | collapsed |
| `n2-derf` | Dynamic erf for LayerNorm | – | – | – | – | stopped at epoch 3 with `n1-dyt`'s early signature |

Health of the runs (server checks, `LADDER_AUDIT.md` section 4):

- Six runs show the exact float16 scale trajectory of a run with no skipped
  step, no NaN, and configs that differ from the control only in the declared
  keys.
- `n1-dyt` lost 8–9 steps to float16 overflow, its loss fell to 0.17 (control
  0.51) and its isotropy reading stayed above 800 (control 34).

Isotropy reading at the end of training against the tracking score: the
reading separates the collapsed run from the healthy ones, and shows no
relation to the score among the six healthy runs.

## 5. What we have learnt

1. **The setup works.** Training is stable in float16, the audit found no
   implementation bug, and the tracking test separates a trained model from an
   untrained one by 6.8 points.
2. **Removing LayerNorm breaks this model as it stands.** LayerNorm sets the
   starting scale the initialisation relies on, keeps gradients inside
   float16's range, and gives every token a centred, fixed-size output, which
   is one of the conditions that keep a JEPA from collapsing. The published
   DyT and Derf results all have a target anchored from outside (labels,
   pixels, or explicit centring); ours is the model's own output.
3. **STRING, as built, cannot add anything.** With frozen frequencies its
   learned basis folds exactly into the attention weights. It tested
   optimisation only, and came out slightly worse.
4. **The screen catches breakages but cannot rank small gains.** It can detect
   effects of about 1.5 points. Published gains for these changes are about
   0.5 points on image classification.
5. **Most of the first ladder was not aimed at the goal.** Five of seven
   changes are generic network tweaks. Only the two RoPE changes touch time,
   and single-frame mode cannot see time.
6. **`f2-relu2` is the one result worth a second look.** It is the cleanest
   comparison (it starts from the control's exact weights). With six
   comparisons, one result of this size appearing by luck is not unusual.

## 6. Known limits of the screen

These apply to every number in section 4.

| Limit | Consequence | What addresses it |
|---|---|---|
| One run per arm; two runs of the same config are not identical (audit L-06) | Training noise is unknown | `a0-control-s2` (running); more seeds with `--seed` |
| 30 test videos; single videos swing by 20 points between runs | A 1-point gap can come from one or two videos | `scripts/compare_tracking.py` resamples by video |
| The same 30 videos are used to choose between changes | Choices can fit this test set rather than tracking in general | Few comparisons; a second seed before believing a win; a held-out benchmark before any claim (not chosen yet) |
| 7,376 training videos, seen about 20 times each | The screen measures learning from little data; a ranking may not hold at scale | The long confirming run; the released-encoder family (section 7) |
| ViT-B from scratch, 9,000 iterations, batch 16 (audit C-10) | Scores are far below the paper's 67; this is "2.1's code on a small recipe", not V-JEPA 2.1 | Same for every arm, so comparisons stay valid |
| Single-frame mode gives every token time position 0 (audit R-07) | It cannot judge a change to the time axis | 16-frame mode |

## 7. What was built on 2026-11-26 (second round)

Everything is behind flags that default to 2.1 behaviour. With all of them at
the default, the training loop still produces the same checkpoint as Meta's
2.1 loop, bit for bit. Nothing here has run on the GPU.

**New experiments**

| Arm | What it tests | Compare with |
|---|---|---|
| `s1-sigreg` | SIGReg (LeJEPA) added to the loss, on the encoder's patch tokens | `a0-control` |
| `s2-visreg` | VISReg added the same way | `a0-control` |
| `t1-future` | Future prediction: for one of the two masks, the encoder sees only the first half of the clip | `a0-control` |
| `t2-fps12` | Frames 12 per second instead of 4 (smaller movements between frames) | `a0-control` |
| `t3-frames` | Whole moments hidden instead of regions: 6 of 8 time-steps hidden and predicted from the other 2 | `a0-control` |
| `t4-whole-clip` | The whole clip is used evenly, instead of mostly its first three quarters | `a0-control` |
| `w0-warm-control` | Encoder started from Meta's released ViT-B; nothing else changed | – (it is a control) |
| `w2-warm-theta-temporal` | Time-axis RoPE change on top of the released encoder | `w0-warm-control` |
| `w3-warm-future` | `t1-future` on top of the released encoder | `w0-warm-control` |
| `w4-warm-fps12` | `t2-fps12` on top of the released encoder | `w0-warm-control` |
| `w5-warm-frames` | `t3-frames` on top of the released encoder | `w0-warm-control` |
| `w6-warm-whole-clip` | `t4-whole-clip` on top of the released encoder | `w0-warm-control` |

Points to keep in mind:

- **Regularisers.** They are added next to the EMA teacher, not in place of
  it, and act on patch tokens because 16 clips are too few to shape a
  distribution. Both choices differ from the papers. The weights (0.1 and
  0.05) are first guesses, set so each adds about a third of the main loss's
  gradient on the encoder at the first step.
- **Released encoder.** Only the encoder can be carried over. Meta's ViT-B was
  made by distillation and its predictor does not fit this training loop, so
  the predictor starts untrained. Two guards protect the encoder from it: no
  encoder update for the first 900 iterations, then a tenth of the learning
  rate. Both are first guesses. The encoder's architecture matches ours
  exactly (158 of 158 weight tensors).

**Tools**

| Tool | What it does |
|---|---|
| `scripts/preflight_arm.py` | One real training step of an arm against the control, on a laptop, in seconds. It flags both norm arms and passes the rest. Run it before any new arm goes to the GPU. |
| `scripts/compare_tracking.py` | Scores against a control with the 95% range, and the gap between two controls as the noise yardstick. |
| `make_ladder_configs.py --local` | Writes configs with the server's real paths into a folder git ignores, so tracked files are never edited by hand. `--seed N` writes repeat runs. |
| Tracking probe `--released` | Scores one of Meta's released encoders with our test. |

**Fixes from the audit**

| Finding | Fix |
|---|---|
| L-20, C-17: `persistent_workers` was never read; workers respawned 19 times a run (about 35 minutes) | The key is read. Kept `false` in the screening recipe so new arms stay comparable with the first ladder. |
| L-10: checkpoints written in place; a failed save only logged | Written to a temporary file and renamed. A failed final save stops the run. |
| L-10: a crashed run did not stop a shell loop | The launcher exits non-zero when a run fails. |
| L-25: server configs were hand-edited copies | `--local` (above). |
| N-22: nothing would have caught the DyT failure early | `preflight_arm.py`. |
| C-22, L-26: no test ran the training loop | `tests/train/`: the real loop, end to end, on CPU. |
| Section 3 of the audit: eight documentation errors | Corrected in the README and code comments. |
| C-04: no gradient clipping | `optimization.clip_grad` (off by default). |

**Verification**

- 259 tests pass, 4 skipped (3 need a GPU, 1 needs Faiss).
- 31 deliberate breakages of the new code were each caught by a test.
- Not verified: anything on the GPU or in float16 for the new paths; loading a
  real released file; `persistent_workers: true` on the server.

**Listed by the audit, not built**

| Item | Why not |
|---|---|
| Learnable RoPE frequencies in place of STRING | New positional design; needs its own review. STRING's result was negative. |
| A norm-free arm that can train | Needs a new initialisation, not a rerun. Parked until the ladder is ranked. |
| SwiGLU with a matched initialisation | Needs a second control with the same initialisation; no sign of a gain so far. |
| Spatial-only RoPE base (`[10000, 100, 100]`) | Config-only; `p1` and `p2` already imply its result. Add it to `ARMS` if wanted. |
| Exact resume (saving RNG state) | A resumed run differs only at the level of a different seed. |
| Recipe changes for the long run (later context-loss ramp, EMA ramp; audit C-11, C-12) | Config-only (`lambda_warmup_frac`, `ema`); to be set when the long run is planned. |

## 8. Back to first principles (2026-11-26)

What tracking needs from the encoder, against what training does today:

| Need | Today | Gap |
|---|---|---|
| Features that separate objects | From scratch on 7,376 videos | Comes mostly from scale: we score 35, Meta's ViT-B 67 |
| The same point keeps its feature over time | The same region is hidden in every frame | Nothing in the task requires following anything through time |
| Small movements | Frames 0.25 s apart, two frames per token | The tracking test runs at 24 fps |
| Small objects | 256 px, 16 px patches | Small objects score about 10, large about 50 |
| A score that sees time | Single-frame mode | It measures still-image features |

**A finding about how clips are fed** (README §9.10, measured on the real mask
generator): the end of every clip is largely unused. About a quarter of all
tokens are neither shown to the encoder nor asked about, nearly all in the
last two of eight time-steps, because Meta's generator cuts each batch's token
lists to the shortest one and always cuts from the end. In the last time-step
the encoder sees 3% of the tokens and is asked about 7%. This is upstream
behaviour, not something we introduced. `t4-whole-clip` changes it: it cuts
at random instead of from the end, so every time-step gets the same share.

**The continuous-time idea** (2026-11-26). A video is a set of samples of a
continuous world, so the model should be asked for moments it was not given:
hide whole frames and predict them from the frames around them, until it can
describe any moment, including ones between recorded frames. `t3-frames` is
the settings-only half of this (whole time-steps hidden). `t1-future` is the
forward-only version. The other half is not built: time positions in seconds
instead of frame counts, and several frame rates per video. It is worth
building only if `t3-frames` or `t1-future` shows a gain in the 16-frame
score. The model predicts features, not pixels; turning those into images
would need a separate decoder.

**Papers checked and set aside**

| Paper | Why it does not apply |
|---|---|
| 2611.24778, Counterfactual World Models | Encoders are frozen; only a small action-conditioned predictor is trained, and the correction is applied at inference. Its "disturbance" is a hidden physical force, not noise in the video. No tracking, masking or frame-rate content. (Main body read; appendices not.) |
| 2610.39302, Platonic Representation Hypothesis with JEPAs | Text and image alignment theory. Abstract only read. |

**Not built: mixed frame rates and a real clock.** Sampling each video at
several frame rates needs an addition to the data loader (one rate per run
today), and it only makes sense with time positions in seconds: positions are
frame counts today, so at mixed rates the same gap would mean different
amounts of real time.

## 9. Next steps, in order

1. **Server:** let `a0-control-s2` finish, then score it in single-frame mode.
2. **Server:** run 16-frame mode on `a0-control` as a memory check, then on
   every finished run.
3. **Here:** rank the first ladder with `compare_tracking.py` and give a keep
   or drop verdict per change.
4. **Decide:** commit and push the second round, pull it on the server.
5. **Server:** download Meta's ViT-B file and score it with `--released`. This
   tells us whether our test agrees with the paper's 67.
6. **Server:** run `s1-sigreg` and `s2-visreg` (about 5 h each). Check
   `reg_r0.csv` after the first epoch.
7. **Server:** run `w0-warm-control` and score it. Only if the score holds,
   run the time-focused arms on it, in this order: `w6-warm-whole-clip`,
   `w5-warm-frames`, `w3-warm-future`, `w4-warm-fps12`, then
   `w2-warm-theta-temporal`. If it does not hold, run the from-scratch twins
   `t4-whole-clip`, `t3-frames`, `t1-future`, `t2-fps12` instead. Score all of
   these in 16-frame mode.
8. **Server, only if `f2-relu2` is still ahead after step 3:** repeat it with
   a second seed.
9. Combine the winners (`PRESET` in `make_ladder_configs.py`) and do one long
   confirming run.

## 10. Server reference

| Item | Value |
|---|---|
| Machine | one Quadro RTX 8000 (48 GB), 96 cores, conda env `modeldeploy` |
| Repo | `/home/user/Documents/dhvanil/RnD/vjepa2` |
| Runs | `/home/user/Documents/dhvanil/RnD/runs/pretrain_3/ladder/<arm>` |
| DAVIS | `/home/user/Documents/dhvanil/RnD/datasets/DAVIS` |
| Logs | `<arm>.log`, `tracking-<arm>.log`, `tracking-clip-<arm>.log` in the repo folder |
| One run | 9,000 iterations, about 5 h |
| Per-run files | `log_r0.csv` (loss), `diag_r0.csv` (isotropy), `reg_r0.csv` (regulariser arms only), `e9/e19/e29.pth.tar`, `latest.pth.tar` |
