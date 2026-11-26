# Ladder audit: theory vs implementation

Date: 2026-11-26. Code audited: commit `1b8271f`.

Six independent reviewers each compared one part of the ablation ladder with its
source paper and with the code that actually runs. They worked on a CPU-only Mac,
could not reach the training server, and changed nothing in the repo. They were
stopped before finishing every planned experiment, so each finding is marked:

- **run** — the reviewer executed the repo's real code and printed the evidence;
- **read** — inferred from reading the code, not executed;
- **server** — can only be settled on the GPU server (see section 4).

The coordinator did not re-run the reviewers' experiments. Paper statements were
mostly read through a summarising fetch tool, not the PDFs, so paper-side quotes
are medium confidence unless stated.

Severity: CRITICAL (the arm's result is invalid or means something else), MAJOR
(changes how the result must be read), MINOR, INFO, VERIFIED (checked, correct).

## 1. Summary

| Arm | Formula matches paper | Wiring correct | Re-run needed | How the result must be read |
|---|---|---|---|---|
| `a0-control` | yes | yes | no, pending server check 1 | "2.1 code on a proxy recipe", not V-JEPA 2.1 as published |
| `p1-theta100` | yes | yes | no | spatial half follows the literature; temporal half is our own hypothesis |
| `p2-theta-temporal` | yes | yes | no | only `clip` mode exercises the change |
| `p3-string` | yes (basis only) | yes | no | a re-parameterisation of q/k weights; not the paper's STRING |
| `f1-swiglu` | yes | yes | no | different start from the control, and an init known to blunt SwiGLU |
| `f2-relu2` | yes | yes | no, pending server check 3 | cleanest paired arm; float16 overflow risk during training unverified |
| `n1-dyt` | yes | yes | **likely, pending server check 2** | DyT dropped into an init that depends on LayerNorm; encoder may be starved of gradient |
| `n2-derf` | yes | yes | **likely, pending server check 2** | same as `n1-dyt` |
| Ladder machinery | – | yes | no | single-variable and fair, but no noise estimate |

Cross-cutting conclusions:

1. No implementation bug was found in any arm's formula or wiring.
2. `n1-dyt` and `n2-derf` are the only arms at risk of being invalid (N-09 to N-11).
3. There is no noise estimate: one run per arm, and identical-seed runs are not
   reproducible (L-06). A second control run is required before any gap is read.
4. The positional arms need `clip` mode (R-07, L-14).
5. The float16 path is correct by reading but unverified on CUDA (C-24).

## 2. Findings by area

### 2.1 `a0-control` and the shared recipe (prefix C)

| ID | Severity | Basis | Finding |
|---|---|---|---|
| C-01 | VERIFIED | run | With all new flags at default, `app/vjepa_3` equals `app/vjepa_2_1`: same `state_dict`, bitwise-equal forward at ViT-B 16f 256px, bitwise-equal checkpoints after 8 steps of both real `train.py` loops (loop parity shown on ViT-L). |
| C-02 | VERIFIED | run | The loss from the real `train_step` equals an independent implementation of the paper's Eq. 1–3 with 4-level deep supervision (max abs diff 1.8e-7). |
| C-03 | VERIFIED | run | Schedules over 9,000 iterations: lr 1e-4 → 2e-4 over 0–899 then constant; weight decay 0.04 / 0; EMA 0.99925; context-loss weight 0 until 450, 0.5 from 900. `ipe_scale` has no effect. |
| C-04 | VERIFIED | run | AdamW, 4 groups that partition all 316 trainable tensors; decay only on ≥2-D non-bias tensors; no gradient clipping (as in 2.1). |
| C-05 | VERIFIED | run | EMA covers 158/158 encoder tensors, runs after `optimizer.step`, in fp32; the model has no buffers. |
| C-06 | VERIFIED | run | The isotropy diagnostic does not change training (bitwise-identical state with it on or off). |
| C-07 | VERIFIED | run | 7,376 videos / batch 16 = 461 batches per pass; 19 refreshes; 20 distinct sampler seeds; no video order is replayed. Holds while the CSV has more than 4,800 videos. |
| C-08 | VERIFIED | run | `a0-control.yaml` equals the base except `folder`; all `model`, `loss`, `mask`, `data_aug` values equal upstream's ViT-B config except the documented `SCREEN` keys. |
| C-09 | VERIFIED | run (CPU stand-in) | GradScaler is called in the correct order. It is required at float16: the unscaled loss gradient is about 1e-8 per element, below float16's smallest value. |
| C-10 | MAJOR | read | The control is not a V-JEPA 2.1 baseline in the paper's sense: ViT-B from scratch (paper: distilled only), 12-block predictor (paper: 24), video only, 7.4k videos, 9k iterations (paper: 135k), batch 16. Arm-vs-control comparisons stay valid. |
| C-11 | MINOR | run | The context-loss ramp (450→900) overlaps the lr warm-up (0→900) and starts when the EMA teacher is 71% random init. Upstream ramps only after the teacher has turned over. |
| C-12 | MINOR | run | EMA 0.99925 is not rescaled for a 9k run: time constant 1,333 iterations (14.8% of the run); teacher more than 50% init until iteration ≈ 924. |
| C-13 | MINOR | run | Five tensors are never trained: the image patch embedding (2), both `img_mod_embed`, and `mask_tokens.1`. A `T == 1` input at evaluation would go through a random patch embedding; the tracking probe uses `T == 2`. |
| C-14 | MINOR | run | Seven arms share the control's initial weights and worker seed; `f1-swiglu` does not. `meta.seed` does not change video order. |
| C-15 | INFO | run | Mask truncation is "minimum over the batch": batch 16 keeps about 525 / 153 context tokens against 435 / 102 at upstream's batch of 96. Same for all arms. |
| C-16 | INFO | run | A clip is 16 frames spanning 3.96 s (3.79 fps at a 24 fps source); short videos are padded with the last frame; sampling is with replacement (63% unique videos per pass). |
| C-17 | INFO | read | `data.persistent_workers` and `loss.reg_coeff` are read by nothing; 24 workers are respawned at each refresh. Timing only. |
| C-18 | INFO | read | Any `latest.pth.tar` already in a run folder is resumed silently. |
| C-19 | INFO | run | The logged loss is L_pred + λ(t)·L_ctx; it steps up between iterations 450 and 900 by construction and must not be used to rank arms. |
| C-20 | INFO | read | Config comments credit "the paper" with upstream-config values (300,000 iterations, lr 6e-4); the paper says 135,000 + 12,000 and 5.25e-4. |
| C-21 | INFO | run | Upstream 2.1 cannot train `vit_base` (`NameError`); the v3 `EMBED_DIM_BY_MODEL_NAME` table is the one behaviour-changing difference on the control path. |
| C-22 | INFO | run | No test executes `train.main`, CUDA autocast, float16 or the GradScaler. |
| C-23 | INFO | read | Encoder levels for ViT-B are blocks 3, 6, 9, 12; the "multi-level predictor" is one linear head 384 → 4×768 on the last predictor block. |
| C-24 | INFO, open risk | server | Which tensors are half precision under CUDA float16 is inferred. On a CPU stand-in float16 lost step 0 to non-finite gradients; bfloat16 did not. Becomes MAJOR for every arm if server check 1 fails. |

### 2.2 `p1-theta100`, `p2-theta-temporal` (prefix R)

| ID | Arm | Severity | Basis | Finding |
|---|---|---|---|---|
| R-01 | both | VERIFIED | run | `rope_theta=10000.0` reproduces 2.1 bit for bit (function, module, full ViT-B forward). |
| R-02 | both | VERIFIED | run | The YAML value reaches all 36 RoPE attention modules (encoder, predictor, target) for scalar and list form. |
| R-03 | p2 | VERIFIED | run | Axis order is `[temporal, height, width]`; element 0 changes the temporal axis only. |
| R-04 | both | VERIFIED | run | `p1` = `p2` + spatial rebase exactly; nothing else is coupled. |
| R-05 | both | VERIFIED | run | No new parameters; seeded init bit-identical to the control. |
| R-06 | both | VERIFIED | run | The frequency formula and rotation match RoFormer per axis slice; logits depend on relative position only. |
| R-07 | p2, p1 (time) | MAJOR | run | `frame` mode has one temporal position, so temporal RoPE is the identity; with equal weights `p2` and the control are the same function there. Read `p2` from `clip` mode. |
| R-08 | both | MINOR | run | "More channels live" holds for the README's >0.2 rad criterion (time 4→8 of 10) but not for wavelength < axis length (time 1→1, space 2→3). The shortest wavelength stays 6.28 positions. |
| R-09 | p1 | MINOR | run | `interpolate_rope` pins the spatial extent to 15 units at every resolution; at 480×848 wavelengths in tokens stretch 1.9× (h) and 3.5× (w). |
| R-10 | both | MINOR | run | The predictor has 5 frequency pairs per axis, not 10; each arm changes encoder and predictor together. |
| R-11 | both | MINOR | read | Base 100 is supported only for from-scratch 2-D axial RoPE (Heo et al.); nothing cited tests it on a temporal axis. |
| R-12 | both | INFO | run (CPU) | float16 table rounding differs slightly between control and rebased axes; 3–4 orders below the effect of the base change. |
| R-13 | both | INFO | read | The base is not stored in the checkpoint; only `params-pretrain.yaml` carries it. Older eval scripts would load these runs at base 10000. |
| R-14 | both | INFO | read | "Warm start: yes" in the arm header means "can be"; the runs start from scratch. |
| R-15 | both | MINOR | read | No test calls `init_video_model`, builds a predictor, or checks axis semantics functionally. |

### 2.3 `p3-string` (prefix S)

| ID | Severity | Basis | Finding |
|---|---|---|---|
| S-01 | MAJOR | run | The learned basis `P` can be folded exactly into each layer's `qkv` projection (difference 5e-15), so the arm and the control have the same function class. The arm tests optimisation dynamics only. |
| S-02 | MAJOR | read | Not the paper's Cayley-STRING: frequencies are frozen (paper: learned), base 10000 (paper: 100), `P` shared across heads (paper default: per head). |
| S-03 | MINOR | run | The README and `string.py` claim `P` lets the t/h/w axes share channels. It does not: the rotation spectrum is unchanged by `P`. |
| S-04 | INFO | run | Adam moves every generator entry by about `lr` per step even with no loss signal; `P` will end far from identity whether or not it helped. |
| S-05 | MINOR | read | Tests do not exercise the train path, float16, the zero-init gradient or the EMA; one test cited in the README does not exist. |
| S-06 | INFO | read | The arm header says "warm start: yes"; the run is from scratch. |
| S-07 | INFO | run (CPU) | In a float16 emulation GradScaler skipped step 0 equally in both arms; STRING is not the overflow bottleneck. |
| S-08 | VERIFIED | run | Cayley formula, antisymmetry, orthogonality and sign convention match the paper's Eq. 10. |
| S-09 | VERIFIED | run | `P` multiplies on the correct side, on q and k only, 64×64 per layer. |
| S-10 | VERIFIED | run | Identity at init: forward and first loss bit-identical to the control; no other parameter's init shifts. |
| S-11 | VERIFIED | run | Non-zero gradient at the zero generator; the generator moves under the real loss path. |
| S-12 | VERIFIED | run | STRING parameters sit alone in a no-weight-decay group that follows the lr schedule. |
| S-13 | VERIFIED | run | The EMA averages the generator; the target's `P` stays exactly orthogonal. |
| S-14 | VERIFIED | run | Translation invariance holds on the real attention module with a non-trivial generator. |
| S-15 | VERIFIED | run (CPU, MPS) | The linear solve runs in float32 with autocast disabled; casting `P` to half costs ≤5e-4 orthogonality error. |
| S-16 | VERIFIED | run | Three batched `[12,64,64]` solves per train step, none per block. |
| S-17 | VERIFIED | run | The predictor keeps plain RoPE and is unaffected. |
| S-18 | VERIFIED | run | A checkpoint written by the real `train.main()` round-trips through the tracking probe at 480×848. |

### 2.4 `f1-swiglu`, `f2-relu2` (prefix F)

| ID | Arm | Severity | Basis | Finding |
|---|---|---|---|---|
| F-01 | f1 | VERIFIED | run | The code computes exactly `fc3(SiLU(fc1 x) * fc2 x)`, Shazeer's SwiGLU (rel. error 5e-15). |
| F-02 | f2 | VERIFIED | run | The code computes exactly `fc2(relu(fc1 x)^2)`; same shapes and parameter count as the control. |
| F-03 | f1 | VERIFIED | run | Hidden width follows the 2/3 rule (3072→2048, 1536→1024); +0.014% / +0.023% parameters; identical multiply-accumulates. |
| F-04 | both | INFO | run | Biases on all FFN linears (Shazeer is bias-free; DINOv2 and upstream have biases); no normalisation inside the FFN (EVA-02 adds one). |
| F-05 | both | VERIFIED | run | The config hop chain is complete; all 12 + 12 blocks are switched. `wide_silu` is `True` only through `train.py`'s default. |
| F-06 | both | INFO | run | The predictor's input-embedding MLP keeps GELU in both arms. |
| F-07 | f1 | MAJOR | run | Same seed does not give a comparable start: no weight matrix matches the control and the data-loader worker seed differs. |
| F-08 | f2 | VERIFIED | run | Same seed gives a bit-identical model and identical RNG state. |
| F-09 | f1 | MAJOR | run | The SwiGLU branch starts at 0.42× (encoder) / about 0.3× (predictor) of the control's FFN-branch magnitude under the inherited init. EVA-02 reports SwiGLU with this init gives no gain (83.9 vs 84.0) and needs xavier-normal (85.0). |
| F-10 | f1 | MINOR | run | `_rescale_blocks` divides `fc2` (the value projection), not the output projection `fc3`. Forward at init is identical either way; gradient roles swap. Inherited from upstream. |
| F-11 | f2 | VERIFIED | run | `f2` starts in a sane regime: FFN-branch rms 1.15× the control (encoder). |
| F-12 | both | INFO | run (MPS) | Under CUDA-style autocast the square runs in float32; overflow happens at the float16 cast into `fc2`, same threshold (pre-activation > 255.9). |
| F-13 | f2 | MINOR, unverified | server | float16 overflow is impossible at init (max pre-activation 7.2) but cannot be excluded during training. The symptom would be a crash on `loss is nan`. |
| F-14 | all | INFO | read | Inherited `trunc_normal_` leaves a handful of weights at exactly −2.0 (100 sigma). |
| F-15 | both | VERIFIED | run | Optimizer groups are correct; nothing is missing or duplicated. |
| F-16 | both | VERIFIED | read | The EMA target is a deep copy; the eval path rebuilds the same FFN and loads strictly. |
| F-17 | both | MINOR | read | No test builds these arms at ViT-B through the train path or under float16. |
| F-18 | f1 | INFO | run | No shipped upstream config sets `use_silu: true`; the SwiGLU branch has no released recipe. |
| F-19 | f2 | VERIFIED | read | The cited negative reference (arXiv 2605.17659) is accurate, but its model is MaxViT-Tiny, and clipping the squared activation recovers most of the gap. |

### 2.5 `n1-dyt`, `n2-derf` (prefix N)

| ID | Severity | Basis | Finding |
|---|---|---|---|
| N-01 | VERIFIED | run | `DynamicTanh` equals the paper's Eq. 2 and the official module (zero numerical difference). |
| N-02 | VERIFIED | run | `DynamicErf` equals the paper's Eq. 10 and the official module (zero numerical difference). |
| N-03 | VERIFIED | run | No `nn.LayerNorm` is left in encoder, predictor or target; the final norm is replaced too, as the papers do. |
| N-04 | INFO | read | The only normalisation left is the loss-side `F.layer_norm` on targets, by design. |
| N-05 | VERIFIED | run | α = 0.5, s = 0, γ = 1, β = 0 survive initialisation in all modules. |
| N-06 | VERIFIED | run | α, s, γ, β all sit in the no-weight-decay groups, as in the official code; the EMA moves them. |
| N-07 | VERIFIED | run | Same seed: all tensors shared with the control are bit-identical. |
| N-08 | VERIFIED | run | A `target_encoder` checkpoint round-trips through the tracking probe's strict loader. |
| N-09 | MAJOR | run | Root cause: V-JEPA's init (weights at std 0.02, zero mask tokens, no added positional vector) relies on LayerNorm to restore scale. With DyT/Derf the predictor output is about 2,000× smaller than the control's and the encoder gradient 8,000–13,000× smaller at init. |
| N-10 | CRITICAL (provisional) | run (fp32) + arithmetic | In float16 the encoder's residual-stream gradients at init sit at the underflow floor: 100% subnormal, 10–30% round to zero (control: 0%). Not measured in float16; duration unknown. |
| N-11 | MAJOR | run | 99.5–99.8% of encoder gradient elements at init are below Adam's eps (1e-8). The encoder's first update is 2.7% (n1) / 4.3% (n2) of lr against 96.8% for the control. Duration unknown. |
| N-12 | VERIFIED | run | Targets at init are healthy: no saturated units; unit variance after standardisation. |
| N-13 | INFO | read | The predictor output is not bounded by γ (a Linear follows `predictor_norm`). The README's claim that neither paper tested this position is wrong for DyT (MAE). |
| N-14 | INFO | read | Only the loss-side `F.layer_norm` guards target scale, in the control too; nothing was found pushing toward collapse. |
| N-15 | INFO | run | `norm_out_type: null` follows `norm_type`. `norm_out_type: layernorm` would be a different, cleaner-interface experiment (not run). |
| N-16 | VERIFIED | run (CPU, MPS) | Under half-precision autocast tanh/erf run in fp32 because α has shape (1,). |
| N-17 | INFO | run | Derf's derivative vanishes much earlier than DyT's (1.4e-4 at z = 3 against 9.9e-3). |
| N-18 | INFO | run | An upstream float16 overflow gives a finite DyT/Derf output but a NaN gradient, so the step is skipped silently; the control would crash instead. |
| N-19 | VERIFIED | run | The patch-embedding scale (RMS 0.78) suits α = 0.5; no blow-up or saturation through 12 blocks at init. |
| N-20 | MINOR | read | `sigreg_raw` is not comparable with the control for these arms; use `sigreg_std`. |
| N-21 | MINOR | read | The probe compares bounded, un-centred features for n1/n2 with normalised features for the rest; score gaps mix representation quality with output geometry. |
| N-22 | MINOR | read | No test compares signal or gradient scale with the control; nothing would have caught N-09 to N-11. |
| N-23 | INFO | read | DyT's DINO run disabled fp16 and clipped gradients; this recipe does neither. Derf reports no float16 training. |

### 2.6 Ladder machinery (prefix L)

| ID | Arm | Severity | Basis | Finding |
|---|---|---|---|---|
| L-01 | all | VERIFIED | run | `make_ladder_configs.py --check` passes; each arm is exactly its declared keys plus `folder`. |
| L-02 | all | VERIFIED | run | Built through the real `train.main` at ViT-B, each arm changes only what it declares in encoder, predictor and target. |
| L-03 | all | VERIFIED | run | Optimizer groups, schedules and data/mask settings are identical across arms except for each arm's own new parameters. |
| L-04 | all but f1 | VERIFIED | run | Six arms start from weights bit-identical to the control and draw the same worker seed. |
| L-05 | f1 | MINOR | run | `f1-swiglu` shares no randomly initialised tensor with the control (same as F-07). |
| L-06 | all | MAJOR | run | One run per arm, and identical-seed runs with worker processes are not reproducible (the mask stream differs). The noise floor is unmeasured. |
| L-07 | all | VERIFIED | run | The diagnostic does not perturb training. |
| L-08 | all | VERIFIED | run | Crash → resume through the real path is bit-identical to an uninterrupted run once the global RNG is restored (8 arms). |
| L-09 | p3 | VERIFIED | run | The resumed `p3-string` run is comparable; it differs only at the level of a different seed. |
| L-10 | all | MINOR | run + read | Checkpoints are written non-atomically; a truncated file fails loudly on resume. A failed save is only logged. A crashed child does not stop the shell loop. |
| L-11 | all | VERIFIED | run | The probe rebuilds the trained architecture: features equal the train-built target encoder exactly for all 8 arms, both modes. |
| L-12 | p1, p2, f2 | MINOR | run | Their checkpoints strict-load into the control architecture without error; only `params-pretrain.yaml` distinguishes them. |
| L-13 | all | VERIFIED | run | The probe scores `target_encoder` (EMA), final layer after the last output norm. |
| L-14 | p1, p2, p3 | MAJOR | run | `frame` mode gives every token temporal position 0 (same as R-07). `clip` mode caveats: frames at 24 fps against 4 fps in training; windows encoded independently. |
| L-15 | n1, n2 | MINOR | run | The scored feature is a bounded output for these arms and a LayerNorm output for the rest (same as N-21). |
| L-16 | all | VERIFIED | run | Probe J and F equal a reference copy of the official DAVIS metrics on 180 mask pairs. |
| L-17 | all | VERIFIED | run | Probe propagation equals DINO's label-propagation algorithm (max diff 2.8e-7). |
| L-18 | all | MINOR | read | Protocol deviations: soft area-averaged first-frame labels, 848 px width, direct upsampling. Equal for all arms. |
| L-19 | all | MINOR | read | `--random-init` is unseeded; the 28.1 floor is one draw for one architecture. |
| L-20 | all | MINOR | run | `data.persistent_workers: true` is never read; workers are respawned every 461 iterations (about 35 minutes per run). |
| L-21 | all | INFO | run | An "epoch" is 300 iterations, not a data pass; sampling ignores `meta.seed`. |
| L-22 | all | INFO | read | Parameter counts differ by at most 0.06%; compute by about 1% at most. |
| L-23 | f1, f2 | INFO | run | `pred_ffn_type` does not change the GELU inside `predictor_embed` (same as F-06). |
| L-24 | all | INFO | read | Python `random` is never seeded with `meta.seed` in the main process. |
| L-25 | all | INFO | server | The repo YAMLs hold placeholder paths; the server ran edited copies that were not audited. |
| L-26 | all | INFO | run | 180 tests pass, 4 skipped. No test goes through `train.main`, checkpoint save/load, resume, or a train-built checkpoint into the probe. |

## 3. Documentation errors to correct

All eight were corrected on 2026-11-26 (README, code comments, the base config
and the generated arm headers).

| Where | Wrong statement | Correct statement | Finding |
|---|---|---|---|
| README §1(a), `string.py:54-56` | STRING lets the axes share channels | The rotation spectrum is unchanged by `P` | S-03 |
| README §1, `p1` YAML comment | "4 of 10 → 8" channels | True for the encoder under a >0.2 rad criterion only; the predictor has 5 pairs | R-08, R-10 |
| README §1, `build_omega` docstring | "convergent evidence" for base 100 | One 2-D heuristic; STRING keeps 10,000 for plain RoPE | R-11 |
| README §6 | `fc2` is the "gate-input" projection | The gate input is `fc1`; `fc2` is the value branch | F-10 |
| README §9, `norms.py` docstring | Neither norm paper tested the output-norm position | DyT's MAE experiment did | N-13 |
| `base-1gpu.yaml`, README §6 | "the paper recipe is 300,000", "the paper's 6e-4" | Those are upstream-config values; the paper says 135,000 + 12,000 and 5.25e-4 | C-20 |
| Arm headers | "warm start: yes" | "can be warm-started"; the ladder runs from scratch | S-06, R-14 |
| README test table, `string.py:222` | cites `test_cls_and_register_scores_unchanged_by_string` | No such test exists | S-05 |

## 4. Server checks

`R=/home/user/Documents/dhvanil/RnD/runs/pretrain_3/ladder`

| # | Settles | Check | Pass condition | Status |
|---|---|---|---|---|
| 1 | C-24, C-09, N-18 | Print `scaler` from each `e*.pth.tar` | `scale` in the tens of thousands or more; not in the low thousands | **pass** for `a0`, `p1`, `p2`, `p3`, `f1`, `f2` (2026-11-26): scale 131072 / 524288 / 1048576 at epochs 10 / 20 / 30, the exact no-skipped-step trajectory. **fail** for `n1-dyt`: 16384 at epoch 10, 512 at epoch 20 (about 8–9 skipped steps, scale 2,048× below the control's) |
| 2 | N-09, N-10, N-11 | Compare the loss of `n1-dyt` / `n2-derf` with `a0-control` over the first 1,800 iterations | Their loss falls with the control's; it does not sit near 0.80–0.83 | **pass** for `n1-dyt`: 0.826 → 0.757 (it 100) → 0.588 (it 200) → 0.453 (it 300); the starvation at init clears within about 200 iterations. `n2-derf` not run |
| 3 | F-13 | `grep -n -i "nan\|AssertionError\|Traceback" f2-relu2.log` | no output | **pass**: no NaN in any `log_r0.csv`, no skipped step in `f2-relu2`. The tracebacks at the end of each finished arm's log are data-loader worker aborts at process exit, after `e29.pth.tar` was written |
| 4 | L-25, L-12 | `diff` each arm's `params-pretrain.yaml` against the control's | only `folder` and the declared keys differ | **pass** for all seven arms that have run |
| 4b | N-14, N-18 | Epoch-mean loss and `sigreg_std` of `n1-dyt` against the control | comparable to the control | **fail** (2026-11-26): `n1-dyt` loss tracks the control to epoch 13, then falls 0.496 → 0.241 between epochs 17 and 19 and reaches 0.173 at epoch 26 (control 0.509). Its `sigreg_std` is 1,160–1,350 through epoch 15 and 800–890 after, against the control's 233 → 34. The loss drop coincides with the skipped float16 steps. Read as feature collapse; to be confirmed by the tracking score. `n2-derf` ran 3 epochs (2026-11-26) and showed the same early signature: `sigreg_std` 219 → 1,170 and `sigreg_raw` 1,017 → 1,419 by epoch 3, iteration 200. Stopped there on this evidence; no loss or scaler reading was taken |
| 4c | L-13 | DAVIS val J&F, `frame` mode, epoch 30, all seven finished arms | – | **done** (2026-11-26). Untrained floor 28.1. `a0-control` 34.9, `p1-theta100` 34.2, `p2-theta-temporal` 35.2, `p3-string` 33.5, `f1-swiglu` 33.9, `f2-relu2` 36.2, `n1-dyt` 18.7. `n1-dyt` is 9.4 below the untrained floor, which confirms the collapse reading in 4b. The six healthy arms span 2.7 points; no noise estimate exists yet, so none is ranked |
| 4d | – | Final isotropy reading (`sigreg_std`, epoch 30, iteration 200) against the `frame` J&F | – | **done** (2026-11-26). `a0-control` 28.8 / 34.9, `p1-theta100` 30.6 / 34.2, `p2-theta-temporal` 37.0 / 35.2, `p3-string` 33.4 / 33.5, `f1-swiglu` 36.6 / 33.9, `f2-relu2` 32.5 / 36.2, `n1-dyt` 1,017 / 18.7. The reading separates the collapsed run from the healthy ones; among the six healthy runs it shows no relation to the tracking score (single reading per run, so weak evidence) |
| 5 | L-06 | Second control run with a different `meta.seed` | gives the noise floor | open |
| 6 | R-07, L-14 | Score all arms in `--mode clip` as well as `frame` | – | open |
| 7 | L-15, N-21 | Re-score `n1`, `n2` and the control on per-token-standardised features | ranking unchanged | open |
| 8 | L-19 | `--random-init` for all 8 arms | arm-specific floors | open |
| 9 | S-04 | Print how far the STRING generators moved (`‖P−I‖`), read against a noise baseline of about 3 | – | open |

## 5. Follow-up

What was built in response to these findings, and what was left, is recorded in
[PROGRESS.md](PROGRESS.md) section 7. In short: L-10, L-20, L-25, N-22, C-22 and
L-26 are fixed or covered by new tooling and tests; the arm redesigns suggested
by S-02, F-09 and N-09 are not built.

## 6. Not verified

- Anything under CUDA float16 (all reviewers worked on CPU or MPS).
- How long the `n1`/`n2` gradient starvation lasts (no training steps were run).
- Whether `f2-relu2` activations approach the float16 limit during training.
- The J and F implementation against the installed official toolkit on real predictions.
- The effect on tracking scores of the bounded output in `n1`/`n2`.
- Paper wording against the PDFs (read through a summarising tool).

The reviewers' scripts and raw outputs are in a temporary scratch folder
(`.../scratchpad/audit/{control,rope,string,ffn,norms,plumbing}`) that can be
cleared by a reboot.
