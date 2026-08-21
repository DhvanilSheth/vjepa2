import torch
import torch.nn as nn
import torchvision


# All token ids here are FLAT indices into a (T_g, H_g, W_g) grid that's been
# flattened the same way PatchEmbed3D flattens it: id = d*H_g*W_g + h*W_g + w.
# These three helpers just undo that flattening (integer div / mod), so an id
# like 137 becomes "frame 2, row 3, col 9" again. Same math as
# Attention.separate_positions in modules.py, kept standalone here so the
# training loop can call it without an attention module attached.

def _get_frame_pos(ids, H_patches=None, W_patches=None, grid_size=None):
    # How many ids make up one full frame (H_g * W_g) -> integer-divide it out.
    if H_patches is None or W_patches is None:
        tokens_per_frame = int(grid_size * grid_size)
    else:
        tokens_per_frame = int(H_patches * W_patches)
    return ids // tokens_per_frame


def _get_height_pos(ids, H_patches=None, W_patches=None, grid_size=None):
    # Remove frame component from ids
    if H_patches is None or W_patches is None:
        tokens_per_frame = int(grid_size * grid_size)
        tokens_per_row = grid_size
    else:
        tokens_per_frame = int(H_patches * W_patches)
        tokens_per_row = W_patches
    frame_ids = _get_frame_pos(ids, H_patches, W_patches, grid_size)
    ids = ids - tokens_per_frame * frame_ids
    # -- now ids is "position within this frame" (0 .. H_g*W_g-1);
    #    divide by row width to get the row (height) index
    return ids // tokens_per_row


def separate_positions(ids, H_patches=None, W_patches=None, grid_size=None):
    # ids: flat token indices, any shape, e.g. (B, N)
    # Returns (frame_ids, height_ids, width_ids), each same shape as ids,
    # i.e. the 3D grid coordinate of every flat index -- this is the
    # "un-flatten" step: flat id -> (d, h, w).
    if H_patches is None or W_patches is None:
        tokens_per_frame = int(grid_size * grid_size)
        tokens_per_row = grid_size
    else:
        tokens_per_frame = int(H_patches * W_patches)
        tokens_per_row = W_patches
    frame_ids = _get_frame_pos(ids, H_patches, W_patches, grid_size)
    # --
    height_ids = _get_height_pos(ids, H_patches, W_patches, grid_size)
    # --
    # Remove frame component from ids (1st term) and height component (2nd term)
    # -- whatever's left is the width (column) index
    width_ids = (ids - tokens_per_frame * frame_ids) - tokens_per_row * height_ids
    return 1.0 * frame_ids, 1.0 * height_ids, 1.0 * width_ids


# WHY THIS FUNCTION EXISTS (check train.py before reading the loop below):
# When predict_all=True, the model is also asked to reconstruct the VISIBLE
# (context/encoder) tokens' target features, not just the masked ones.
# Not every context token is equally useful for that: a context token sitting
# right next to the masked hole is doing real work (it's the token the model
# would lean on to predict the hole), while one far away from any mask is
# trivial to reconstruct and shouldn't count as much.
# This function measures, for every context token, its 3D grid distance to
# the NEAREST masked/target token. train.py then uses that distance as
# `1 / d` to weight the context-reconstruction loss per token (see
# `d_weights` in train.py's loss_fn) -- closer to the hole = higher weight.
def compute_mask_distance(masks_pred, masks_enc, grid_size, offset_context_loss):
    # masks_pred: [fpc][mask] where each mask is [B, N_pred]  (target/masked token ids)
    # masks_enc:  [fpc][mask] where each mask is [B, N_enc]   (visible/context token ids)
    # fpc = "frames per clip" grouping / mask config index -- there can be
    # several independent mask samples per clip (see MaskCollator), so this
    # is a nested loop: outer over mask-group, inner over each mask in it.
    distances = []
    for masks_pred_i, masks_enc_i in zip(masks_pred, masks_enc):
        row_distances = []
        for masks_pred_ij, masks_enc_ij in zip(masks_pred_i, masks_enc_i):
            N_enc_tokens = masks_enc_ij.shape[1]

            # Step 1. Decode every context token's flat id -> (d, h, w) grid coord.
            d_enc, h_enc, w_enc = separate_positions(
                masks_enc_ij, grid_size=grid_size
            )  # (BS, N_enc)

            # Step 2. Same for every target/masked token.
            d_pred, h_pred, w_pred = separate_positions(
                masks_pred_ij, grid_size=grid_size
            )  # (BS, N_pred)
            pred = torch.stack([d_pred, h_pred, w_pred], dim=-1)  # (BS, N_pred, 3)

            # Step 3. For each context token, in turn, find its Euclidean
            # distance (in grid-coordinate units, NOT pixels) to every target
            # token, then keep only the closest one.
            enc_distances = []
            for enc_token in range(N_enc_tokens):
                enc_position = torch.stack(
                    [d_enc[:, enc_token], h_enc[:, enc_token], w_enc[:, enc_token]],
                    dim=-1,
                ).unsqueeze(
                    1
                )  # (BS, 1, 3) -- this one context token's (d,h,w), batched
                dist = torch.cdist(enc_position, pred, p=2)  # (BS, N_enc)
                dmin, argmin = dist.min(dim=-1)  # nearest target token's distance
                if offset_context_loss:
                    # Optional rescale so the distance doesn't grow just because
                    # you trained at a larger grid_size than the 16x16 default.
                    coeff = grid_size // 16  # Which is the default value of grid_size
                    dmin = dmin * (1.0 / coeff)
                dmin = dmin**0.5  # We want that it decreases less agressive
                enc_distances.append(dmin)
            # Stack per-token distances back into one (BS, N_enc) tensor --
            # this is what train.py divides into (1 / d) to weight the loss.
            enc_distances = torch.stack(enc_distances, dim=-1).squeeze()  # (BS, N_enc)
            row_distances.append(enc_distances)
        distances.append(row_distances)
    return distances
