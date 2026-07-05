# V-JEPA 2.1 — Implementation Notes
## Part I: Foundational Model Utilities (`app/vjepa_2_1/models/utils/`)

**Author:** Code study prepared for review
**Subject:** Meta FAIR V-JEPA 2.1 (release `0.0.2`, commit `45d025f`)
**Scope of this part:** the lowest layer of the model stack — patch embedding, positional
embedding, and the transformer building blocks — from which the V-JEPA 2.1 encoder and
predictor are assembled.

---

### Abstract

V-JEPA 2.1 is a self-supervised video representation model built on a Vision Transformer (ViT)
encoder and a lightweight predictor, trained with a joint-embedding predictive objective. This
document is the first in a planned bottom-up series that reads the V-JEPA 2.1 source code from
its primitives upward to the training loop. Part I covers the three files in
`app/vjepa_2_1/models/utils/` that supply the architectural primitives: `patch_embed.py`
(tokenization of images and video), `pos_embs.py` (sinusoidal positional encodings), and
`modules.py` (feed-forward networks, self-attention, rotary position embeddings, and the
transformer block). Each component is presented with its mathematical formulation, a mapping to
the implementing code, and a worked numerical example. Where a design choice is specific to
V-JEPA 2.1, this is identified explicitly.

---

### Document conventions

- **Code references** are given as `file.py:line` (e.g., `modules.py:97`). Paths are relative to
  `app/vjepa_2_1/models/utils/` unless stated otherwise.
- **Tensor shapes** are written in brackets, e.g. `[B, N, D]`. The symbols used throughout are:

  | Symbol | Meaning |
  |:------:|---------|
  | `B`    | batch size |
  | `T`    | number of input frames |
  | `H`, `W` | spatial height and width (pixels) |
  | `C`    | input channels (3 for RGB) |
  | `N`    | number of tokens in a sequence |
  | `D`, `d` | token embedding dimension |
  | `h`    | feed-forward hidden dimension |

- **Running example.** Unless noted otherwise, examples use a single 16-frame RGB clip at
  256×256 resolution, patch size 16, tubelet size 2, and embedding dimension `D = 768`
  (the ViT-Base configuration).

---

## 1. Overview

### 1.1 Architectural context

A Vision Transformer does not operate on pixels directly; it operates on a *sequence of token
vectors*. Producing that sequence, annotating it with positional information, and processing it
through stacked attention-and-feed-forward layers are exactly the responsibilities of the three
utility files documented here. The encoder (`vision_transformer.py`) and predictor
(`predictor.py`), covered in later parts, are assembled almost entirely from these primitives.

The data flow is summarized in Figure 1.

```
Figure 1. Data flow through the model-utility layer.

   raw input  [B, C, T, H, W]
        |
        |  patch_embed.py            §2   tokenization (strided convolution)
        v
   token sequence  [B, N, D]
        |
        |  pos_embs.py (+ token)     §3   absolute sinusoidal positions   (used when use_rope=False)
        |  -- or --
        |  RoPE (inside attention)        relative rotary positions       (V-JEPA 2.1 default)
        v
   +--------------------------------------------------+
   |  Transformer Block  x depth          §4          |
   |    x = x + Attention( LayerNorm(x) )             |
   |    x = x + FeedForward( LayerNorm(x) )           |
   +--------------------------------------------------+
        |
        v
   contextualized features  [B, N, D]
```

### 1.2 Files covered in Part I

| File | Lines | Responsibility | Section |
|------|------:|----------------|:-------:|
| `patch_embed.py` | 72 | Convert images/video into a sequence of patch tokens | §2 |
| `pos_embs.py` | 95 | Generate fixed sinusoidal positional encodings (1-D/2-D/3-D) | §3 |
| `modules.py` | 544 | Feed-forward networks, self-attention, rotary embeddings, transformer block | §4 |

---

## 2. Patch embedding (`patch_embed.py`)

### 2.1 Purpose

The patch-embedding layer maps a raw image or video tensor to a sequence of fixed-dimensional
token vectors. This is the interface between pixel space and the transformer.

### 2.2 Mechanism: convolution as patch extraction

Rather than explicitly cropping patches and projecting them, the implementation uses a single
convolution whose kernel size equals its stride. Such a convolution tiles the input into
*non-overlapping* patches and applies a learned linear projection to each, in one operation.

**Listing 1 — 2-D image tokenizer (`patch_embed.py:30`).**
```python
self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)  # :38
x = self.proj(x).flatten(2).transpose(1, 2)                                             # :42
```

For the running example (a 256×256 image, `patch_size = 16`, `D = 768`), the shape evolution is
given in Table 1.

**Table 1 — Shape evolution in `PatchEmbed.forward`.**

| Step | Operation | Shape |
|------|-----------|-------|
| input | — | `[B, 3, 256, 256]` |
| `self.proj(x)` | `Conv2d`, 16×16 kernel, stride 16 | `[B, 768, 16, 16]` |
| `.flatten(2)` | flatten the 16×16 spatial grid | `[B, 768, 256]` |
| `.transpose(1, 2)` | move tokens to dimension 1 | `[B, 256, 768]` |

The result, `[B, N, D] = [B, 256, 768]`, is a sequence of 256 patch tokens, each a 768-dimensional
vector.

### 2.3 The video tokenizer

`PatchEmbed3D` (`patch_embed.py:46`) generalizes the construction by one dimension, using a
`Conv3d` with kernel and stride `(tubelet_size, patch_size, patch_size)`. Each token therefore
summarizes a *tubelet* — a block spanning `tubelet_size` frames and a `patch_size × patch_size`
spatial region. Table 2 traces the running example.

**Table 2 — Shape evolution in `PatchEmbed3D.forward` (16 frames, `tubelet_size = 2`).**

| Step | Shape |
|------|-------|
| input `[B, C, T, H, W]` | `[B, 3, 16, 256, 256]` |
| `self.proj(x)` (`Conv3d`) | `[B, 768, 8, 16, 16]` |
| `.flatten(2).transpose(1, 2)` | `[B, 2048, 768]` |

This yields `N = 8 × 16 × 16 = 2048` tokens. Because each token aggregates two consecutive
frames, a degree of short-range motion information is captured at the tokenization stage.

### 2.4 Relevance to V-JEPA 2.1: a unified image/video tokenizer

One of the four innovations of V-JEPA 2.1 is *multi-modal tokenization*: a single model trained
jointly on images and video. This is enabled at the lowest level by instantiating `PatchEmbed3D`
twice — once with `tubelet_size = 2` for video, and once with `tubelet_size = 1` for still
images (the `patch_embed_img` path in the encoder). With `tubelet_size = 1`, the 3-D convolution
on a single-frame input reduces to ordinary per-frame patching. Consequently, images and video
yield token sequences of identical structure and can traverse the same transformer stack; only
the tokenizer differs.

**Remark.** The file also defines `AudioPatchEmbed` (`patch_embed.py:13`) for spectrogram inputs.
It is not used by the vision model and is omitted from further discussion.

---

## 3. Positional embeddings (`pos_embs.py`)

### 3.1 Purpose

Self-attention (§4.2) is permutation-invariant: it produces identical outputs under any
reordering of its input tokens. Positional information must therefore be supplied explicitly.
This file constructs *fixed* (non-learned) sinusoidal positional encodings in one, two, and
three dimensions.

**Note.** The default V-JEPA 2.1 configuration sets `use_rope = true`, which instead injects
position through rotary embeddings within attention (§4.3). The encodings in this file are the
fallback used when `use_rope = false`. They are nonetheless foundational, as rotary embeddings
reuse the same frequency construction.

### 3.2 The one-dimensional sinusoidal encoding

All higher-dimensional encodings are built from a single primitive
(`pos_embs.py:77`), the encoding introduced by Vaswani et al. [1]. For a position `p` and an
embedding dimension `D`, define a bank of `D/2` geometrically spaced angular frequencies:

$$\omega_i = 10000^{-\,i/(D/2)}, \qquad i = 0, 1, \dots, \tfrac{D}{2}-1.$$

The encoding concatenates the sine and cosine of `p` scaled by each frequency:

$$\mathrm{PE}(p) = \big[\,\sin(p\,\omega_0),\dots,\sin(p\,\omega_{D/2-1}),\ \cos(p\,\omega_0),\dots,\cos(p\,\omega_{D/2-1})\,\big].$$

**Listing 2 — frequency bank and encoding (`pos_embs.py:84–94`).**
```python
omega = 1.0 / 10000 ** (np.arange(D // 2) / (D / 2.0))   # frequencies, high -> low
out   = np.einsum("m,d->md", pos, omega)                 # (positions, D/2)
emb   = np.concatenate([np.sin(out), np.cos(out)], axis=1)
```

High frequencies vary rapidly with position and encode fine-grained location; low frequencies
vary slowly and encode coarse location. The construction has two properties relevant to
attention: distinct positions receive distinct encodings, and a fixed offset between positions
corresponds to a linear transformation of their encodings.

### 3.3 Two- and three-dimensional encodings

The 2-D encoding (`pos_embs.py:43`, for images) partitions the embedding budget equally between
the row and column axes: each axis is encoded by the 1-D primitive on `D/2` dimensions, and the
two results are concatenated.

The 3-D encoding (`pos_embs.py:11`, used for video) extends this to depth/time `d`, height `h`,
and width `w`. The allocation of the embedding budget among the three axes is controlled by
`uniform_power`:

**Listing 3 — axis budget allocation (`pos_embs.py:26–37`).**
```python
if not uniform_power:                  # default
    d_embed_dim = embed_dim // 2       # the temporal axis receives half the budget
    h_embed_dim = w_embed_dim = embed_dim // 4
else:                                  # V-JEPA 2.1 configs set uniform_power = true
    h_embed_dim = w_embed_dim = d_embed_dim = int(np.ceil(embed_dim / 6) * 2)
pos_embed = np.concatenate([emb_d, emb_h, emb_w], axis=1)[:, :embed_dim]
```

- With `uniform_power = False` (the default), the temporal axis receives half of the embedding
  dimension and each spatial axis a quarter, biasing the representation toward temporal position.
- With `uniform_power = True` (the V-JEPA 2.1 setting), all three axes are weighted equally. For
  `D = 768`, each axis receives `ceil(768/6) × 2 = 256` dimensions, totaling 768; the trailing
  slice `[:, :embed_dim]` removes any surplus introduced by the ceiling.

The optional class-token branches prepend a zero row; the V-JEPA 2.1 encoder does not use a class
token, so these branches are inactive in practice.

### 3.4 Integration

The encoder evaluates `get_3d_sincos_pos_embed` once at initialization, registers the result as a
fixed buffer, and adds it to the patch tokens immediately after embedding — but only when
`use_rope = False`. Under the default rotary configuration, no additive term is used and position
is introduced inside attention, as described next.

---

## 4. Transformer building blocks (`modules.py`)

This file provides the components from which every transformer layer is constructed. Table 3
summarizes them and indicates which lie on the default V-JEPA 2.1 code path.

**Table 3 — Components of `modules.py`.**

| Component | Definition | Role | On default 2.1 path |
|-----------|:----------:|------|:-------------------:|
| `MLP` | `:71` | Standard feed-forward network | When `use_silu = False` |
| `SwiGLUFFN` | `:97` | Gated feed-forward network | Yes, when `use_silu = True` |
| `DropPath` | `:57` | Stochastic-depth regularization | Yes (no-op when rate = 0) |
| `Attention` | `:305` | Standard multi-head self-attention | Fallback (`use_rope = False`) |
| `rotate_queries_or_keys` | `:14` | Rotary transformation of Q/K | Yes |
| `RoPEAttention` | `:128` | Self-attention with rotary positions | Yes (default) |
| `Block` | `:355` | One transformer layer | Yes |
| `CrossAttention`, `CrossAttentionBlock` | `:459`, `:497` | Cross-attention | No (downstream use) |
| `Lambda_LinearWarmupHold` | `:523` | Scalar schedule for the dense-loss weight | Yes (used by `train.py`) |

### 4.1 Feed-forward networks

Within a transformer layer, attention exchanges information *between* tokens, whereas the
feed-forward network (FFN) transforms *each token independently*. Two FFN variants are provided.

#### 4.1.1 Standard FFN (`MLP`, `modules.py:71`)

The standard FFN is two linear layers separated by a fixed nonlinearity (GELU [7]):

$$\mathrm{MLP}(x) = W_2\,\phi(W_1 x),$$

where `W_1` expands the dimension from `d` to a hidden size `h` (typically `h = 4d`) and `W_2`
projects back to `d`. Its limitation is that the same fixed nonlinearity `φ` is applied to every
feature; the layer cannot modulate features conditionally on the input. The gated variant
addresses this.

#### 4.1.2 Gated FFN (`SwiGLUFFN`, `modules.py:97`)

`SwiGLUFFN` implements the SwiGLU feed-forward network of Shazeer [3], which is now standard in
large transformer models (e.g., PaLM, LLaMA). It combines three ideas — a smooth activation, a
gating mechanism, and a parameter-budget adjustment — developed below.

**(a) Activation: SiLU.** The Sigmoid Linear Unit [6, 7] is

$$\mathrm{SiLU}(x) = x\,\sigma(x), \qquad \sigma(x) = \frac{1}{1+e^{-x}}.$$

It is smooth and non-monotonic. For large positive `x`, `σ(x) → 1` and `SiLU(x) ≈ x` (the input
passes through); for large negative `x`, `σ(x) → 0` and `SiLU(x) → 0` (the input is suppressed).
It attains a minimum of approximately `−0.278` near `x ≈ −1.278`. The factor `σ(x)` behaves as a
smooth gate in `(0, 1)`, so the unit may be described as *self-gating*.

**(b) Gating: the Gated Linear Unit.** A GLU [2] forms two linear projections of the input and
lets one modulate the other elementwise:

$$\mathrm{GLU}(x) = (W_1 x) \odot \sigma(W_2 x),$$

where `⊙` denotes the elementwise (Hadamard) product. The term `σ(W_2 x)` is a *learned,
input-dependent gate*: a per-feature multiplier whose value depends on the input through its own
weights `W_2`. This introduces a multiplicative interaction between two learned views of the
input, enabling conditional behaviour (e.g., emphasizing one feature only when another is
present) that a purely additive MLP cannot easily represent.

**(c) SwiGLU.** Replacing the sigmoid gate with SiLU and adding an output projection gives the
SwiGLU feed-forward network:

$$\mathrm{FFN}_{\mathrm{SwiGLU}}(x) = W_3\big(\,\mathrm{SiLU}(W_1 x)\ \odot\ (W_2 x)\,\big).$$

The correspondence to the implementation is given in Table 4.

**Table 4 — Mapping of SwiGLU to code (`modules.py:121–125`), shapes for `d = 768`, `h = 2048`.**

| Mathematical term | Code | Output shape |
|-------------------|------|--------------|
| `W_1 x` | `x1 = self.fc1(x)` | `[B, N, 2048]` |
| `W_2 x` | `x2 = self.fc2(x)` | `[B, N, 2048]` |
| `SiLU(W_1 x) ⊙ (W_2 x)` | `hidden = F.silu(x1) * x2` | `[B, N, 2048]` |
| `W_3(·)` | `self.fc3(hidden)` | `[B, N, 768]` |

The gate is `F.silu(x1)`, a per-feature multiplier applied to `x2`. Table 5 illustrates its
effect on a single channel with content value `x2 = 3.0` as the gate input `x1` varies.

**Table 5 — Gating behaviour for fixed content `x2 = 3.0`.**

| Gate input `x1` | `SiLU(x1)` | `hidden = SiLU(x1)·x2` | Effect |
|:---------------:|:----------:|:----------------------:|--------|
| `+4.0` | `3.93` | `11.8` | gate open; content amplified |
| `+2.0` | `1.76` | `5.28` | gate open |
| `0.0`  | `0.00` | `0.00` | gate closed; content blocked |
| `−2.0` | `−0.24` | `−0.71` | gate nearly closed |
| `−4.0` | `−0.07` | `−0.21` | gate closed; content suppressed |

The same content value is amplified, passed, or suppressed depending on a second learned signal —
the conditional routing a fixed activation cannot provide.

**(d) Parameter budget (`wide_silu`).** SwiGLU uses three weight matrices rather than two. With a
common hidden size `h`, a standard FFN has `2dh` parameters whereas SwiGLU has `3dh` — a 50%
increase. To compare the two at equal cost, the SwiGLU hidden size is reduced so that
`3 d h' = 2 d h`, i.e. `h' = (2/3) h`:

**Listing 4 — hidden-size reduction and hardware alignment (`modules.py:111–115`).**
```python
swiglu_hidden_features = int(2 * hidden_features / 3)            # 2/3 reduction
align_as = 8                                                     # round up to a multiple of 8
swiglu_hidden_features = (swiglu_hidden_features + align_as - 1) // align_as * align_as
```

For `d = 768` and a nominal `h = 4d = 3072`, the reduced size is `h' = 2048`. The parameter
counts then match exactly: the standard FFN uses `2 × 768 × 3072 = 4{,}718{,}592` parameters and
SwiGLU uses `3 × 768 × 2048 = 4{,}718{,}592`, since `2 × 3072 = 3 × 2048 = 6144`. The gating
mechanism is thus obtained at no additional parameter cost. The final rounding raises `h'` to the
nearest multiple of 8 for efficient execution on tensor-core hardware.

**Implementation note.** The constructor assigns `self.act = act_layer()` (`modules.py:118`), but
`forward` calls `F.silu` directly; `self.act` is therefore unused, and the activation is fixed to
SiLU regardless of the `act_layer` argument.

#### 4.1.3 Stochastic depth (`DropPath`, `modules.py:57`)

`DropPath` implements stochastic depth [8]: during training it randomly zeroes an entire residual
branch for a subset of samples, which regularizes deep networks by training over an implicit
ensemble of shallower sub-networks. It is a no-op when the drop rate is zero, as in most
V-JEPA 2.1 configurations.

### 4.2 Self-attention (`Attention`, `modules.py:305`)

#### 4.2.1 Formulation

Each token produces a query, a key, and a value vector. A token's output is a weighted average of
all value vectors, where the weights are the softmax-normalized similarities between that token's
query and every key. For queries `Q`, keys `K`, and values `V`:

$$\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\!\left(\frac{QK^{\top}}{\sqrt{d_k}}\right) V,$$

with `d_k` the per-head dimension. The scaling by `1/√d_k` prevents the dot products from growing
large enough to saturate the softmax. Attention is computed independently in `num_heads` parallel
subspaces (multi-head attention), allowing different heads to attend to different relationships.

#### 4.2.2 Implementation

For the running example, `x = [B, N, D] = [2, 2048, 768]`, `num_heads = 12`, and
`d_k = 768 / 12 = 64`.

**Listing 5 — projection to multi-head Q/K/V (`modules.py:331–336`).**
```python
qkv = self.qkv(x)                              # Linear(768 -> 2304): Q, K, V combined
   .reshape(B, N, 3, self.num_heads, C // self.num_heads)   # [2, 2048, 3, 12, 64]
   .permute(2, 0, 3, 1, 4)                                  # [3, 2, 12, 2048, 64]
q, k, v = qkv[0], qkv[1], qkv[2]                            # each [2, 12, 2048, 64]
```

A single linear layer produces `Q`, `K`, and `V` together (output width `2304 = 3 × 768`); the
reshape separates them. The attention computation (`modules.py:345–349`) follows the formula
above and finally recombines the heads into a `[B, N, D]` tensor, which is mixed by an output
projection.

In production the module calls `F.scaled_dot_product_attention` (`modules.py:338`), PyTorch's
fused FlashAttention kernel [9]. This computes the identical result without materializing the
`N × N` attention matrix, which is necessary for the long sequences arising from video.

The standard `Attention` module carries no positional information; this is supplied by the
rotary mechanism described next.

### 4.3 Rotary position embeddings (RoPE)

#### 4.3.1 Motivation

Additive encodings (§3) supply *absolute* position. Rotary Position Embedding (RoPE) [4] instead
supplies *relative* position: it rotates the query and key vectors by an angle proportional to
their position, so that the attention score between two tokens depends only on the difference of
their positions. Relative position is well suited to vision and video and generalizes more
gracefully across input resolutions.

#### 4.3.2 The rotation (`rotate_queries_or_keys`, `modules.py:14`)

The vector is partitioned into consecutive coordinate pairs, and each pair is rotated by an angle
`θ = p · ω` (with `ω` the frequency bank of §3.2). For a pair `(a, b)`:

$$a' = a\cos\theta - b\sin\theta, \qquad b' = b\cos\theta + a\sin\theta.$$

**Listing 6 — rotation by coordinate pairs (`modules.py:39–44`).**
```python
y = x.unflatten(-1, (-1, 2)); y1, y2 = y.unbind(dim=-1)   # split into (even, odd) pairs
y = torch.stack((-y2, y1), dim=-1).flatten(-2)            # the perpendicular component
out = (x * emb_cos) + (y * emb_sin)                       # 2-D rotation per pair
```

*Example.* The pair `(1.0, 0.0)` rotated by `θ = 90°` becomes `(0.0, 1.0)`. Two tokens at
different positions rotate by different angles; their relative angle is what persists in the dot
product `q · k`. Class and register token positions are excluded from rotation
(`modules.py:24–26`), as they have no spatial coordinate.

#### 4.3.3 Three-dimensional factorization (`RoPEAttention`, `modules.py:128`)

A video token has three coordinates — time, height, and width — so each attention head's
dimension is divided into three blocks, each rotated by a different coordinate:

**Listing 7 — per-axis block sizes (`modules.py:155–157`).**
```python
self.d_dim = self.h_dim = self.w_dim = int(2 * ((head_dim // 3) // 2))
```

For `head_dim = 64` (the value in all V-JEPA 2.1 model sizes), each block has 20 dimensions.
Dimensions `[0:20)` are rotated by the temporal coordinate, `[20:40)` by height, and `[40:60)` by
width; the remaining four dimensions are left unrotated (`modules.py:276–280`).

The mapping from a flat token index to its `(time, height, width)` coordinates is computed by
`separate_positions` (`modules.py:187`):

$$
\text{frame} = \left\lfloor \frac{\mathrm{id}}{HW} \right\rfloor,\quad
\text{height} = \left\lfloor \frac{\mathrm{id} - HW\cdot\text{frame}}{W} \right\rfloor,\quad
\text{width} = \mathrm{id} - HW\cdot\text{frame} - W\cdot\text{height}.
$$

*Example.* For a 16×16 spatial grid (`HW = 256`), token index 300 maps to `frame = 1`,
`height = 2`, `width = 12`; its query and key are rotated by these three coordinates in the three
respective blocks.

#### 4.3.4 Masking and position (a V-JEPA-specific point)

In the joint-embedding predictive setting, the encoder processes only the *visible* subset of
tokens, which are no longer at contiguous indices. The `mask` argument to `RoPEAttention.forward`
is therefore the list of *original grid indices* of the retained tokens; `separate_positions` is
applied to it so that each retained token is rotated according to its true position in the full
grid (`modules.py:215–217`). This is how positional information is preserved through masking.

#### 4.3.5 Resolution interpolation (`interpolate_rope`, `modules.py:227`)

V-JEPA 2.1 pre-trains at 256-pixel resolution (a 16×16 grid) and evaluates at 384 pixels
(a 24×24 grid). Coordinates `0…23` lie outside the range seen during training, so they are
rescaled into the training range:

**Listing 8 — coordinate rescaling (`modules.py:232–233`).**
```python
h_mask = h_mask * (self.pretrained_grid_size - 1) / (H_patches - 1)
w_mask = w_mask * (self.pretrained_grid_size - 1) / (W_patches - 1)
```

With `pretrained_grid_size = 256/16 = 16`, the mapping `[0, 23] → [0, 15]` aligns the rotary
wavelengths at evaluation with those at training. Only the spatial coordinates are interpolated;
the temporal coordinate is handled separately. After rotation, the attention computation is the
same scaled-dot-product attention as in §4.2.

### 4.4 The transformer block (`Block`, `modules.py:355`)

A `Block` is the unit replicated `depth` times to form the encoder (12 times for ViT-Base, 24 for
ViT-Large, and so on). It consists of two residual sub-layers in the pre-normalization
arrangement:

**Listing 9 — block forward pass (`modules.py:451–452`).**
```python
x = x + self.drop_path(self.attn(self.norm1(x)))   # token mixing (attention)
x = x + self.drop_path(self.mlp(self.norm2(x)))    # per-token transformation (FFN)
```

Three points are noteworthy:

1. **Pre-normalization.** Layer normalization is applied *before* each sub-layer, the standard
   arrangement for stable training of deep transformers.
2. **Residual connections.** Each sub-layer computes an additive update; gradients propagate
   directly through the residual path, which permits very deep stacks (up to 48 layers in the
   largest model).
3. **Configuration-driven substitution.** The constructor selects `RoPEAttention` or `Attention`
   according to `use_rope` (`modules.py:382`), and `SwiGLUFFN` or `MLP` according to whether the
   activation is SiLU (`modules.py:413`).

The `mode` argument to `Block.forward` (`modules.py:437`) is accepted for call-signature
compatibility but is not used within the block.

### 4.5 Auxiliary components

**Cross-attention (`modules.py:459`, `:497`).** `CrossAttention` and `CrossAttentionBlock`
implement attention in which queries are drawn from one sequence and keys/values from another
(`forward(self, q, x)`). This is the mechanism used for attentive pooling. These modules are not
part of the encoder/predictor self-attention stack and are not exercised during pre-training.

**Dense-loss schedule (`Lambda_LinearWarmupHold`, `modules.py:523`).** This is a scalar schedule,
not a network module. It governs `λ`, the weight applied to the *dense context loss* — the term
in the V-JEPA 2.1 objective that predicts visible (context) tokens in addition to masked targets:

$$
\lambda(t) =
\begin{cases}
0, & t < t_{\text{start}} \\[2pt]
\lambda_{\max}\,\dfrac{t - t_{\text{start}}}{t_{\text{end}} - t_{\text{start}}}, & t_{\text{start}} \le t < t_{\text{end}} \\[8pt]
\lambda_{\max}, & t \ge t_{\text{end}}
\end{cases}
$$

with defaults `t_start = 15{,}000` and `t_end = 30{,}000` iterations. The training loop queries
`lambda_sched.value(epoch · ipe + itr)`. The schedule allows the model to first learn the
standard masked-prediction objective and to introduce the denser objective gradually, which
improves training stability. This component is the first concrete link to the dense predictive
loss analyzed in a later part.

---

## 5. Summary

The three files in `app/vjepa_2_1/models/utils/` provide a complete and self-contained set of
architectural primitives:

- **Tokenization** (`patch_embed.py`) converts images and video to a common token-sequence
  representation via strided convolution, with a unified image/video tokenizer underpinning
  V-JEPA 2.1's multi-modal training.
- **Positional encoding** (`pos_embs.py`) supplies fixed sinusoidal positions; in the default
  configuration this role is taken over by rotary embeddings within attention.
- **Transformer blocks** (`modules.py`) provide the feed-forward networks (standard and gated),
  multi-head self-attention, the three-dimensional rotary position mechanism, and the
  pre-normalized residual block, together with a schedule for the dense-loss weight.

Two design decisions in this layer are specific to V-JEPA 2.1 and recur in later parts: the
unified image/video tokenizer (§2.4), and the rotary position mechanism with masking-aware
positions and resolution interpolation (§4.3). The dense-loss schedule (§4.5) anticipates the
training objective documented subsequently.

**Planned continuation.** Part II will document `masks_dist.py` (the distance weighting of the
dense loss) and `vision_transformer.py` (assembly of the encoder, including the intermediate-layer
outputs that implement *deep self-supervision*). Part III will cover the predictor and the
training loop.

---

## References

[1] A. Vaswani et al. "Attention Is All You Need." *NeurIPS*, 2017.

[2] Y. Dauphin et al. "Language Modeling with Gated Convolutional Networks." *ICML*, 2017.

[3] N. Shazeer. "GLU Variants Improve Transformer." arXiv:2002.05202, 2020.

[4] J. Su et al. "RoFormer: Enhanced Transformer with Rotary Position Embedding."
arXiv:2104.09864, 2021.

[5] A. Dosovitskiy et al. "An Image Is Worth 16×16 Words: Transformers for Image Recognition at
Scale." *ICLR*, 2021.

[6] S. Elfwing, E. Uchibe, K. Doya. "Sigmoid-Weighted Linear Units for Neural Network Function
Approximation in Reinforcement Learning." *Neural Networks*, 2018. (See also P. Ramachandran,
B. Zoph, Q. Le, "Searching for Activation Functions," arXiv:1710.05941, 2017.)

[7] D. Hendrycks, K. Gimpel. "Gaussian Error Linear Units (GELUs)." arXiv:1606.08415, 2016.

[8] G. Huang et al. "Deep Networks with Stochastic Depth." *ECCV*, 2016.

[9] T. Dao et al. "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness."
*NeurIPS*, 2022.

[10] L. Mur-Labadia et al. "V-JEPA 2.1: Unlocking Dense Features in Video Self-Supervised
Learning." arXiv:2603.14482, 2026.
