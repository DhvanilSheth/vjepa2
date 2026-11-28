# V-JEPA 3: progress record

Last updated: 2026-11-27.

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
| Second control run (training noise) | done 2026-11-27 (`a0-control-s2`, seed 1239): **32.3**, which is 2.6 points below `a0-control` |
| Meta's released ViT-B scored with our test | done 2026-11-27: **68.9** (paper: 67.0) |
| Tracking scores, 16-frame mode | done 2026-11-27 for the 7 healthy runs; it fits in memory |
| Ranking against noise | done 2026-11-27: no change is separable from training noise; `n1-dyt` is clearly worse (section 4) |
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
| Meta's released ViT-B, as released | – | 68.9 | +34.0 | – | – | the ceiling we can see; paper reports 67.0 |
| `n1-dyt` | Dynamic Tanh for LayerNorm | 18.7 | −16.2 | −20.3 to −12.4 | 3 / 58 | collapsed |
| `n2-derf` | Dynamic erf for LayerNorm | – | – | – | – | stopped at epoch 3 with `n1-dyt`'s early signature |

Meta's released ViT-B (2026-11-27, single-frame mode): J&F 68.9 (J 66.0, F
71.9); small objects 47.3, medium 71.2, large 78.7. Our from-scratch control
scores about 10 / 30 / 50 on the same three groups. Two things follow: our
test agrees with the paper to within 2 points, so it can be trusted as a
yardstick; and the loader for released files works on a real file.

### Ranking against training noise (2026-11-27)

The second control, same config with seed 1239, scored 32.3 in single-frame
mode and 32.4 in 16-frame mode. The gap between the two controls is 2.6
points in both modes. This is one pair of runs, so 2.6 is itself a rough
estimate. `scripts/compare_tracking.py` verdicts, each arm against
`a0-control`:

| Run | Single-frame | vs control | 16-frame | vs control | 95% range (16-frame) | Verdict |
|---|---|---|---|---|---|---|
| `p2-theta-temporal` | 35.2 | +0.3 | 35.9 | +0.9 | −0.1 to +2.0 | not separable |
| `f2-relu2` | 36.2 | +1.4 | 35.5 | +0.5 | −0.8 to +1.7 | inside training noise |
| **`a0-control`** | **34.9** | – | **35.0** | – | – | – |
| `p1-theta100` | 34.2 | −0.6 | 34.5 | −0.5 | −1.9 to +0.8 | not separable |
| `p3-string` | 33.5 | −1.4 | 34.1 | −0.9 | −2.1 to +0.0 | inside training noise |
| `f1-swiglu` | 33.9 | −1.0 | 33.1 | −1.9 | −3.4 to −0.4 | inside training noise |
| `a0-control-s2` | 32.3 | −2.6 | 32.4 | −2.6 | – | the noise yardstick |
| `n1-dyt` | 18.7 | −16.2 | not run | – | – | clear: worse |

Two readings:

- **No change in the first ladder is kept or dropped on its score.** Every
  healthy arm sits inside the gap between two runs of the unchanged model.
  `n1-dyt` and `n2-derf` are dropped because they collapse.
- **Showing the model 16 frames instead of one does not help it track.** Every
  run scores the same in both modes to within a point. These from-scratch
  models make no visible use of time, which is the gap the time-focused arms
  (section 8) are aimed at.

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
4. **The screen catches breakages but cannot rank small gains.** Two runs of
   the unchanged model differ by 2.6 points, so a single run per arm can only
   show effects of about 3 points or more. Published gains for these changes
   are about 0.5 points on image classification.
5. **Most of the first ladder was not aimed at the goal.** Five of seven
   changes are generic network tweaks. Only the two RoPE changes touch time,
   and single-frame mode cannot see time.
6. **`f2-relu2` did not hold up.** Its +1.4 in single-frame mode is inside
   training noise and falls to +0.5 in 16-frame mode. No second seed is
   planned for it.
7. **The from-scratch models do not use time.** 16-frame mode scores the same
   as single-frame mode for every run (section 4).

## 6. Known limits of the screen

These apply to every number in section 4.

| Limit | Consequence | What addresses it |
|---|---|---|
| One run per arm; two runs of the same config are not identical (audit L-06) | Training noise measured once: 2.6 points between two controls | More seeds with `--seed` before believing any gap under about 3 points |
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
| `n3-dyt-sigreg`, `n4-derf-sigreg` | `n1-dyt` / `n2-derf` with SIGReg (added 2026-11-27) | `a0-control`, `n1-dyt` |
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

### 8.1 Masking by content, with real time (design in progress, 2026-11-27)

Nothing in this subsection is built. It records the logic agreed so far.

**Aim.** One masking rule that is correct for an image and for video at any
frame rate, with difficulty rising over training. Time understanding is the
aim; tracking is the yardstick.

**Three kinds of hidden patch**, by where its content can be found:

| Kind | Where its content is | What the model must do | Value |
|---|---|---|---|
| Copy | same place, another time, unchanged | nothing | none; with an absolute-error loss (`loss_exp: 1.0`) it pushes as hard as a hard patch, so it adds noise |
| Follow | another time, a different place | find the same thing and carry it over | this is tracking |
| Imagine | nowhere visible | infer it from the surroundings | teaches what objects are |

V-JEPA's mask (same region for the whole clip) has no Copy, is mostly
Imagine, and has Follow only where something crosses the mask's edge.

**The rule.** Never ask for a patch that looks the same as a patch that is
shown at the same place. Stated on pairs (shown patch, asked patch), so slow
drift needs no grouping. Copy is then impossible for any amount of masking.
A still place costs one shown patch however many frames sampled it, so extra
frames from a high frame rate add nothing to ask about.

**One description for every input.** A sample is a set of frames with real
times in seconds; a patch is (time, place). An image is one frame: all time
differences are zero, so the time axis has no effect, and every hidden patch
is Imagine. Each frame is turned into tokens on its own.

**Two dials, both scheduled over training progress.** `rho`: share hidden
completely (Imagine), in large blocks; active for images and video. `G`: the
longest time in seconds between a hidden changing patch and the nearest
moment shown (Follow); no effect on images.

**Earlier formulation, superseded.** A per-time-step hide fraction drawn from
a Beta distribution and shuffled over time was checked numerically. Its
"leak" (share of hidden patches whose place is visible at another time) was
0.48 to 0.90 with the first-guess settings, against 0 for V-JEPA. That
measure counted places, not content, and the design did not remove Copy.

**Open items.** (1) Deciding "unchanged": in progress, see below. (2) A gap
in seconds is not the whole difficulty; speed matters too. (3) Every clip in
a batch must show the same number of tokens. (4) The one-GPU loop cannot mix
images and video. (5) Single-frame tokens change the input layer for video.
(6) DAVIS alone cannot judge time understanding; a direct test is needed.

**Item 1, the change signal.** `scripts/measure_change.py` compares two
frames patch by patch after three steps, each needed on synthetic scenes with
a known answer: low-pass (noise), per-patch mean removal and gain fit
(brightness), smallest difference over shifts of up to `tol` pixels (camera
shake). Raw pixel difference fails: a 1-pixel shake (0.052) reads as more
change than a real 8-pixel pan (0.039). Dividing by each patch's contrast
also fails, because it amplifies noise in flat patches. On the synthetic
scenes the working signal separates the two groups by a factor of about 2,
which is too narrow to fix a threshold without real video.

**Measured on DAVIS val (2026-11-27, 30 videos, shift tolerance 1 px).**

- *No natural threshold.* At a 0.25 s gap the sizes of change form one
  spread-out group, not two. "Unchanged" is a matter of degree.
- *The signal lands on objects.* At 0.25 s and threshold 0.012, 28% of
  background patches read as unchanged against 4% of object patches. In the 6
  videos with a still camera (background changed in 1% to 30% of patches),
  objects read as changed in 95% to 100%.
- *Most DAVIS videos have a moving camera.* In 19 of 30, at least 75% of the
  background reads as changed. The signal cannot tell camera motion from
  object motion.
- *Copy is a smaller share than assumed at our time scale.* About 19% of all
  patches fall below 0.0105 at 0.25 s. At 1/24 s it is far larger: 50% of
  background and 27% of object patches at threshold 0.012. So the rule
  matters most at high frame rates.

The earlier assumption "most of a video is still" does not hold on DAVIS.

Re-run with a 3-pixel tolerance: the picture does not change. Background
unchanged at 0.25 s rises only from 28% to 35%, and the same 19 videos stay at
70% or more changed. So it is real camera motion, not hand-held drift.

**Decision (2026-11-27): camera motion is not a target.** Mapping a camera's
motion is a solved problem elsewhere (SLAM, 3D reconstruction), so the model's
effort should go to things that move on their own. The rule becomes: never ask
for a patch that looks the same as a shown patch at the same place *after
undoing the camera's motion*. A still camera is the case of zero motion, so
Copy and "explained by the camera" are one class. The kinds are then:
explained by the camera (not asked), Follow (moves on its own), Imagine.

**When both the camera and the object move.** Exact separation is not always
possible from pixels: an object that fills the frame, an object moving along
the camera's path, and a camera moving through depth are each ambiguous. The
rule does not need the separation. It needs to know what is trivial to
predict. So the definition adopted on 2026-11-27 is:

- *Question asked of each patch:* is it explained by the one motion most of
  the frame shares? If yes, it is not asked for.
- *Soft, not yes or no:* a patch is asked for with a weight between 0 and 1,
  in proportion to how much the shared motion fails to explain it. DAVIS
  showed no natural threshold.

| Case | Explained | Asked |
|---|---|---|
| Camera pans, small object moves | background | the object |
| Object fills the frame, camera follows | the object's body | background, and parts that move differently |
| Object moves along the camera's path | far background | the object and near still things |
| Camera moves through depth | almost nothing | almost everything (today's behaviour) |
| Water, leaves, crowds | nothing | all of it |

Nothing trivial is ever asked, and the worst case equals today. What is not
promised: that the asked patches are always the object. That needs depth.

`scripts/measure_change.py` estimates the shared motion as one shift for the
whole frame (`shared_shift`: coarse on the low-pass frames, then exact to the
pixel; decided by the median) and measures change again after undoing it. It
reports how many videos fall in each case, and on DAVIS whether the weight
(`unexplained_weight`) lands on the annotated objects. A `--videos` mode
samples the training videos, which have no masks. A single shift does not
describe zoom, rotation or depth.

**Measured with the shared motion undone (2026-11-27, threshold 0.012).**

| | DAVIS val (30 videos) | Training videos (200 of 7,376) |
|---|---|---|
| Patches changed at 0.25 s, as measured | background 72%, object 96% | 61% |
| ... after undoing the shared motion | background 40%, object 95% | 43% |
| Patches changed at one frame, after undoing | background 17%, object 72% | 20% |
| Median shared shift at 0.25 s | 22 px | 4 px |
| Videos: still / explained / partly / not explained | 1 / 1 / 18 / 10 | 30 / 32 / 110 / 28 |

- *The rule points at objects, modestly.* On DAVIS objects are 30% of the
  patches. Their share of the weight rises from 34-39% as measured to 46-58%
  after undoing the shared motion. About half the weight still lands on
  background.
- *Undoing the shared motion removes background, not objects.* Background
  changed falls from 72% to 40%; object changed stays at 95%.
- *Training videos are calmer than DAVIS.* At a 0.25 s gap 57% of their
  patches are explained by the shared motion, and 80% at a one-frame gap. A
  mask that hides whole moments would spend that share of its targets on
  trivial patches.
- *One shift is too crude for a third of DAVIS and a seventh of the training
  videos* ("not explained"); over half of both are only partly explained.
- *Still no two groups* after undoing the shared motion. The soft weight
  stands.
- The training-video reader works on real files (0 of 200 unreadable).

### 8.2 Continuing from Meta's encoder: `w0-warm-control` (scored 2026-11-28)

Meta's ViT-B encoder, our training loop unchanged, 30 epochs of the screening
recipe (`freeze_encoder_iters: 900`, `encoder_lr_scale: 0.1`).

| Checkpoint | Single-frame J&F | Small / medium / large | 16-frame J&F |
|---|---|---|---|
| Meta, as released | 68.9 | 47.3 / 71.2 / 78.7 | 62.3 |
| after 10 epochs | 68.3 | 47.0 / 70.1 / 78.4 | not measured |
| after 30 epochs | 66.0 | 44.7 / 67.6 / 76.2 | 60.5 |

- The encoder is not wrecked, but it erodes: 0.6 points after a third of the
  run, 2.9 by the end. The recipe pulls a strong encoder towards what this
  small recipe can support (about 35 from scratch).
- Arms on this track are compared with 66.0, not 68.9. Training noise on this
  track is not measured.
- 16-frame mode is 5.5 points below single-frame for this model. Meta's own
  model shows the same drop (68.9 to 62.3), so it is not caused by our
  training. The losses are largest on fast-motion videos. Two candidate
  causes, not yet separated: the test feeds frames 1/24 s apart to a model
  trained on wider spacing, and each token covers two frames, so one feature
  map has to serve two frames in which a fast object sits in two places.
  Until that is checked, 16-frame mode is safe for comparing runs with each
  other and not for judging how well a model uses time.
- In 16-frame mode the erosion from our training is 1.8 points (62.3 to 60.5).

`w0-warm-control` started on the server at 12:14 on 2026-11-27; its log
confirms the encoder was initialised from Meta's file.

### 8.3 The four plain time-focused runs (scored 2026-11-28)

From scratch, screening recipe, epoch 30. The two controls set the band.

| Run | What it changes | Single-frame | 16-frame |
|---|---|---|---|
| `t4-whole-clip` | uses the whole clip evenly | 36.6 | 34.6 |
| `a0-control` | nothing | 34.9 | 35.0 |
| `t1-future` | context from the first half only | 34.3 | 33.3 |
| `t2-fps12` | 12 frames per second instead of 4 | 33.0 | 33.6 |
| `t3-frames` | hides whole moments | 33.1 | 31.2 |
| `a0-control-s2` | nothing, second seed | 32.3 | 32.4 |

- None beats the control by the 3 points the screen can resolve.
- `t3-frames` is the lowest, and in 16-frame mode it is below both controls.
  That fits the measurement in 8.1 (over half of its targets are trivial in
  the training videos), but one run inside the noise does not prove it.
- `t2-fps12` is the only run that scores higher with 16 frames than with one
  (+0.6); the others lose 1 to 2 points. Too small to read alone.

Verdicts from `compare_tracking.py` against `a0-control` (noise 2.6):

| Run | Single-frame | 95% range | 16-frame | 95% range | Verdict |
|---|---|---|---|---|---|
| `t4-whole-clip` | +1.7 | +0.8 to +2.8 | −0.4 | −2.4 to +1.9 | inside noise |
| `t1-future` | −0.6 | −1.6 to +0.5 | −1.7 | −2.7 to −0.8 | inside noise |
| `t2-fps12` | −1.9 | −3.5 to −0.6 | −1.4 | −2.9 to +0.0 | inside noise |
| `t3-frames` | −1.8 | −3.7 to −0.3 | −3.8 | −5.8 to −1.2 | clear: worse (16-frame); 13 objects better, 48 worse |

`t3-frames` is the baseline the content-based rule of 8.1 has to beat.

### 8.4 The research gap, and the first build against it (2026-11-28)

**The gap in V-JEPA's masking study** (its section E.4, 26 runs on a ViT-B,
all judged on action classification): shortening the mask in time made scores
worse, the task was called trivial, and the mask was extended over the whole
clip. Three things were left open:

1. Trivial is a property of each patch, not of the mask. Covering the whole
   clip removes the copyable patches and, with them, the patches where
   something moved.
2. Every hidden patch counts the same in the loss, and with an absolute-error
   loss each pushes equally hard.
3. The camera is ignored; and nothing was scored on a per-patch task.

Their observation holds in our hands: `t3-frames` is clearly worse than the
control. What is new is the explanation (57% of its targets can be copied)
and a way to test it. LeJEPA does not use this masking at all: it compares
crops of an image.

**Built** (`README.md` §9.11): `app/vjepa_3/change.py` and
`target_weight: unexplained` on a mask entry. Each hidden token of a
whole-time-step mask is weighted by how much the shown time-steps fail to
explain it once the shared motion is undone. Arms `t5-frames-weighted`
(against `t3-frames`) and `w7-warm-frames-weighted` (against
`w5-warm-frames`). 320 tests pass; 22 deliberate breakages each caught; the
default path is still bit-for-bit V-JEPA 2.1; the pre-flight passes.

**Not verified:** nothing has run on the GPU. The weights cost about 0.6 s of
one data-loading worker per clip; whether training slows down is unknown.

## 9. Next steps, in order

Steps done on 2026-11-27: second control scored, 16-frame mode run on every
healthy run, first ladder ranked (section 4), Meta's ViT-B scored.

1. **Server:** score Meta's ViT-B in 16-frame mode, if not done.
   Also run `python scripts/measure_change.py --davis <DAVIS>` (CPU only,
   a few minutes) and send the output; it settles item 1 of section 8.1.
2. **Server:** write local configs with `--pretrained`, then run this chain
   (about 5 h each): `w0-warm-control`, `t4-whole-clip`, `t3-frames`,
   `t1-future`, `t2-fps12`.
3. **Server:** score `w0-warm-control` at `e9.pth.tar` and at the end. If it
   stays near Meta's 68.9, run the arms on the released encoder:
   `w6-warm-whole-clip`, `w5-warm-frames`, `w3-warm-future`, `w4-warm-fps12`,
   `w2-warm-theta-temporal`. If it drops, tune `freeze_encoder_iters` and
   `encoder_lr_scale` first.
4. **Here:** read the time-focused arms in 16-frame mode against both
   controls. An arm counts only if it beats `a0-control` by more than the 2.6
   point gap; anything that does gets a second seed.
5. **Server, next:** `t5-frames-weighted` (about 5 h, possibly more: watch
   the iteration time). Compare with `t3-frames` in both modes; check
   `weight_r0.csv` after the first epoch.
6. **Server, later:** `s1-sigreg` and `s2-visreg`. Check `reg_r0.csv` after
   the first epoch.
   Then `n3-dyt-sigreg`, only if `s1-sigreg` is healthy. Pre-flight on
   2026-11-27 (one step, float32, synthetic clips): SIGReg lifts the encoder
   gradient of the DyT model from 0.019x to 0.21x of the control's, but the
   predictor gradient stays at 0.0002x, because SIGReg acts on the encoder
   output only. The predictor's starting scale is a separate problem that
   SIGReg does not address.
6. Combine the winners (`PRESET` in `make_ladder_configs.py`) and do one long
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
