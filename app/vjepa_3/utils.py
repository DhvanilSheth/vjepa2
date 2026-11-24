# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import sys

import app.vjepa_3.models.predictor as vit_pred
import app.vjepa_3.models.vision_transformer as video_vit
import torch
import torch.nn.functional as F
import yaml
from app.vjepa_3.wrappers import MultiSeqWrapper, PredictorMultiSeqWrapper
from src.utils.checkpoint_loader import robust_checkpoint_loader
from src.utils.schedulers import (
    CosineWDSchedule,
    LinearDecaySchedule,
    WarmupCosineSchedule,
)

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


def normalize_and_concat(tensor, embed_dim):
    """Split tensor into 4 chunks of size embed_dim along the last axis,
    apply LayerNorm to each chunk, then concatenate back."""
    chunks = [
        F.layer_norm(tensor[:, :, i * embed_dim : (i + 1) * embed_dim], (embed_dim,))
        for i in range(4)
    ]
    return torch.cat(chunks, dim=2)


def normalize_nested(nested, embed_dim):
    """Apply normalize_and_concat recursively over nested lists."""
    return [
        [[normalize_and_concat(z, embed_dim) for z in inner] for inner in outer]
        for outer in nested
    ]


def build_eval_args(
    model_name,
    patch_size,
    tubelet_size,
    num_frames,
    logging_folder,
    checkpoint,
    write_tag,
    eval_cfg_paths,
    uniform_power=False,
    use_sdpa=False,
    clip_duration=None,
    use_silu=False,
    wide_silu=True,
    tag=None,
):
    """
    Helper function to parse the pre-training configs to construct the
    evaluation configs, return as a list of eval configs.
    """
    import warnings

    if eval_cfg_paths is None:
        logger.info("No evaluations specified!")
        return

    eval_nodes = None
    eval_tasks_per_node = None
    args_eval = []
    for i, f in enumerate(eval_cfg_paths):
        with open(f, "r") as y_file:
            _args = yaml.load(y_file, Loader=yaml.FullLoader)
            _tag = _args.get("tag", "")
            _args["tag"] = f"{tag}-{_tag}"
            _nodes = _args.get("nodes", None)
            _tasks = _args.get("tasks_per_node", 8)
            eval_nodes = _nodes if eval_nodes is None else eval_nodes
            eval_tasks_per_node = (
                _tasks if eval_tasks_per_node is None else eval_tasks_per_node
            )
            if (eval_nodes != _nodes) or (eval_tasks_per_node != _tasks):
                warnings.warn(
                    "Configs for online evals must use same number of nodes for slurm-batch processing"
                )

            _args["pretrain"] = {}
            _args["pretrain"]["model_name"] = model_name
            _args["pretrain"]["patch_size"] = patch_size
            _args["pretrain"]["tubelet_size"] = tubelet_size
            _args["pretrain"]["uniform_power"] = uniform_power
            _args["pretrain"]["use_sdpa"] = use_sdpa
            _args["pretrain"]["clip_duration"] = clip_duration
            _args["pretrain"]["use_silu"] = use_silu
            _args["pretrain"]["wide_silu"] = wide_silu
            _args["pretrain"]["frames_per_clip"] = num_frames
            _args["pretrain"]["folder"] = logging_folder
            _args["pretrain"]["checkpoint"] = checkpoint
            _args["pretrain"]["write_tag"] = write_tag

            args_eval += [_args]

    return eval_nodes, eval_tasks_per_node, args_eval


def load_checkpoint(
    r_path,
    encoder,
    predictor,
    target_encoder,
    opt,
    scaler,
    is_anneal=False,
):
    logger.info(f"Loading {r_path}")
    checkpoint = robust_checkpoint_loader(r_path, map_location=torch.device("cpu"))

    epoch = 0
    if not is_anneal:
        epoch = checkpoint["epoch"]

    pretrained_dict = checkpoint["encoder"]
    for k, v in encoder.state_dict().items():
        if k not in pretrained_dict:
            logger.info(f'key "{k}" could not be found in loaded state dict')
        elif pretrained_dict[k].shape != v.shape:
            logger.info(
                f'key "{k}" is of different shape in model and loaded state dict'
            )
            pretrained_dict[k] = v
    msg = encoder.load_state_dict(pretrained_dict, strict=False)
    logger.info(f"loaded pretrained encoder from epoch {epoch} with msg: {msg}")

    pretrained_dict = checkpoint["predictor"]
    for k, v in predictor.state_dict().items():
        if k not in pretrained_dict:
            logger.info(f'key "{k}" could not be found in loaded state dict')
        elif pretrained_dict[k].shape != v.shape:
            logger.info(
                f'key "{k}" is of different shape in model and loaded state dict'
            )
            pretrained_dict[k] = v
    msg = predictor.load_state_dict(pretrained_dict, strict=False)
    logger.info(f"loaded pretrained predictor from epoch {epoch} with msg: {msg}")

    if target_encoder is not None:
        pretrained_dict = checkpoint["target_encoder"]
        for k, v in target_encoder.state_dict().items():
            if k not in pretrained_dict:
                logger.info(f'key "{k}" could not be found in loaded state dict')
            elif pretrained_dict[k].shape != v.shape:
                logger.info(
                    f'key "{k}" is of different shape in model and loaded state dict'
                )
                pretrained_dict[k] = v
        msg = target_encoder.load_state_dict(pretrained_dict, strict=False)
        logger.info(
            f"loaded pretrained target encoder from epoch {epoch} with msg: {msg}"
        )

    try:
        opt.load_state_dict(checkpoint["opt"])
    except ValueError:
        # V-JEPA 3: was a bare print() in 2.1.  Promoted to logger.warning
        # because this fires whenever the param-group layout changes -- e.g.
        # warm-starting a use_string=True run from a use_string=False
        # checkpoint, which adds STRING groups.  Training continues with FRESH
        # Adam moments; losing that message in stdout costs days of confusion.
        logger.warning(
            "Optimizer groups mismatch; reinitializing optimizer. "
            "Adam moment estimates from the checkpoint have been DISCARDED. "
            "Expected when the param-group layout changed (e.g. enabling "
            "use_string); unexpected otherwise -- check your config."
        )
    if scaler is not None:
        scaler.load_state_dict(checkpoint["scaler"])
    logger.info(f"loaded optimizers from epoch {epoch}")
    logger.info(f"read-path: {r_path}")
    del checkpoint

    return (
        encoder,
        predictor,
        target_encoder,
        opt,
        scaler,
        epoch,
    )


def init_video_model(
    device,
    patch_size=16,
    max_num_frames=16,
    tubelet_size=2,
    model_name="vit_base",
    crop_size=224,
    pred_depth=6,
    pred_num_heads=None,
    pred_embed_dim=384,
    uniform_power=False,
    use_mask_tokens=False,
    num_mask_tokens=2,
    zero_init_mask_tokens=True,
    use_sdpa=False,
    use_rope=False,
    use_silu=False,
    use_pred_silu=False,
    wide_silu=False,
    is_causal=False,
    pred_is_causal=False,
    use_activation_checkpointing=False,
    return_all_tokens=False,
    chop_last_n_tokens=0,
    init_type="default",
    img_temporal_dim_size=None,
    n_registers=0,
    n_registers_predictor=0,
    has_cls_first=False,
    interpolate_rope=False,
    rope_theta=10000.0,
    rope_temporal_scale=1.0,
    use_string=False,
    use_string_predictor=False,
    string_share_across_layers=False,
    string_init_std=0.0,
    modality_embedding=False,
    # -- V-JEPA 3 FFN / norm variants.  ffn_type is per-model (mirrors
    #    use_silu / use_pred_silu); the norm keys apply to both models.
    ffn_type=None,
    pred_ffn_type=None,
    norm_type=None,
    norm_out_type=None,
):
    encoder = video_vit.__dict__[model_name](
        img_size=crop_size,
        patch_size=patch_size,
        num_frames=max_num_frames,
        tubelet_size=tubelet_size,
        uniform_power=uniform_power,
        use_sdpa=use_sdpa,
        use_silu=use_silu,
        wide_silu=wide_silu,
        use_activation_checkpointing=use_activation_checkpointing,
        is_causal=is_causal,
        use_rope=use_rope,
        init_type=init_type,
        img_temporal_dim_size=img_temporal_dim_size,
        n_registers=n_registers,
        has_cls_first=has_cls_first,
        interpolate_rope=interpolate_rope,
        rope_theta=rope_theta,
        rope_temporal_scale=rope_temporal_scale,
        use_string=use_string,
        string_share_across_layers=string_share_across_layers,
        string_init_std=string_init_std,
        modality_embedding=modality_embedding,
        ffn_type=ffn_type,
        norm_type=norm_type,
        norm_out_type=norm_out_type,
    )
    encoder = MultiSeqWrapper(encoder)
    predictor = vit_pred.__dict__["vit_predictor"](
        img_size=crop_size,
        use_mask_tokens=use_mask_tokens,
        patch_size=patch_size,
        num_frames=max_num_frames,
        tubelet_size=tubelet_size,
        embed_dim=encoder.backbone.embed_dim,
        predictor_embed_dim=pred_embed_dim,
        depth=pred_depth,
        num_heads=(
            encoder.backbone.num_heads if pred_num_heads is None else pred_num_heads
        ),
        uniform_power=uniform_power,
        num_mask_tokens=num_mask_tokens,
        zero_init_mask_tokens=zero_init_mask_tokens,
        use_rope=use_rope,
        use_sdpa=use_sdpa,
        is_causal=pred_is_causal,
        use_silu=use_pred_silu,
        wide_silu=wide_silu,
        use_activation_checkpointing=use_activation_checkpointing,
        return_all_tokens=return_all_tokens,
        chop_last_n_tokens=chop_last_n_tokens,
        n_registers=n_registers_predictor,
        has_cls_first=has_cls_first,
        interpolate_rope=interpolate_rope,
        rope_theta=rope_theta,
        rope_temporal_scale=rope_temporal_scale,
        use_string=use_string_predictor,
        string_share_across_layers=string_share_across_layers,
        string_init_std=string_init_std,
        modality_embedding=modality_embedding,
        img_temporal_dim_size=img_temporal_dim_size,
        ffn_type=pred_ffn_type,
        norm_type=norm_type,
        norm_out_type=norm_out_type,
    )
    predictor = PredictorMultiSeqWrapper(predictor)

    encoder.to(device)
    predictor.to(device)
    logger.info(encoder)
    logger.info(predictor)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"Encoder number of parameters: {count_parameters(encoder)}")
    logger.info(f"Predictor number of parameters: {count_parameters(predictor)}")

    return encoder, predictor


def init_opt(
    is_anneal,
    encoder,
    predictor,
    iterations_per_epoch,
    start_lr,
    ref_lr,
    warmup,
    num_epochs,
    use_radamw=False,
    wd=1e-6,
    final_wd=1e-6,
    final_lr=0.0,
    mixed_precision=False,
    ipe_scale=1.25,
    betas=(0.9, 0.999),
    eps=1e-8,
    zero_init_bias_wd=True,
):
    # -- V-JEPA 3: STRING parameters must NOT be weight-decayed.
    #
    #    V-JEPA 2.1 split the groups by `("bias" not in n) and (len(p.shape) != 1)`.
    #    A CayleySTRING weight has shape [num_layers, head_dim, head_dim] --
    #    ndim 3, no "bias" in the name -- so it would land in the DECAY group.
    #    Decay pulls W -> 0, hence S -> 0, hence P -> I: it would actively
    #    unlearn the basis STRING exists to learn.  CosineWDSchedule
    #    (src/utils/schedulers.py:90-92) then ramps that decay every step for
    #    any group without WD_exclude.
    #
    #    The exclusion is UNCONDITIONAL -- deliberately not wired to
    #    zero_init_bias_wd, which governs a different decision (whether
    #    zero-initialised biases get decayed).
    #
    #    Also note these are LISTS, not the generators V-JEPA 2.1 used.
    #    Generator param groups are consumed on first iteration, which makes
    #    them impossible to inspect or assert over.
    def _is_string_param(name):
        return "string.weight" in name

    def _is_decay_param(name, param):
        if _is_string_param(name):
            return False
        return ("bias" not in name) and (len(param.shape) != 1)

    def _is_nodecay_param(name, param):
        if _is_string_param(name):
            return False
        return ("bias" in name) or (len(param.shape) == 1)

    def _select(module, predicate):
        return [p for n, p in module.named_parameters() if predicate(n, p)]

    encoder_string = _select(encoder, lambda n, p: _is_string_param(n))
    predictor_string = _select(predictor, lambda n, p: _is_string_param(n))

    param_groups = [
        {"params": _select(encoder, _is_decay_param)},
        {"params": _select(predictor, _is_decay_param)},
        {
            "params": _select(encoder, _is_nodecay_param),
            "WD_exclude": zero_init_bias_wd,
            "weight_decay": 0,
        },
        {
            "params": _select(predictor, _is_nodecay_param),
            "WD_exclude": zero_init_bias_wd,
            "weight_decay": 0,
        },
    ]
    # Only add STRING groups when STRING is enabled; an empty param group is
    # legal but clutters the optimizer state and the checkpoint diff.
    for string_params in (encoder_string, predictor_string):
        if string_params:
            param_groups.append(
                {
                    "params": string_params,
                    "WD_exclude": True,
                    "weight_decay": 0,
                }
            )

    # The three predicates must PARTITION the parameters: every parameter in
    # exactly one group.  A silent gap here means a tensor never gets a
    # gradient step, which is close to invisible at training time.
    _n_grouped = sum(len(g["params"]) for g in param_groups)
    _n_total = len(list(encoder.parameters())) + len(list(predictor.parameters()))
    assert _n_grouped == _n_total, (
        f"init_opt param groups do not partition the model: grouped "
        f"{_n_grouped} of {_n_total} tensors"
    )

    if use_radamw:
        from src.utils.adamw import AdamW as RAdamW

        logger.info("Using Rescaled-AdamW")
        optimizer = RAdamW(param_groups, betas=betas, eps=eps)
    else:
        logger.info("Using AdamW")
        optimizer = torch.optim.AdamW(param_groups, betas=betas, eps=eps)

    if not is_anneal:
        scheduler = WarmupCosineSchedule(
            optimizer,
            warmup_steps=int(warmup * iterations_per_epoch),
            start_lr=start_lr,
            ref_lr=ref_lr,
            final_lr=final_lr,
            T_max=int(ipe_scale * num_epochs * iterations_per_epoch),
        )
    else:
        scheduler = LinearDecaySchedule(
            optimizer,
            ref_lr=ref_lr,
            final_lr=final_lr,
            T_max=int(ipe_scale * num_epochs * iterations_per_epoch),
        )
    wd_scheduler = CosineWDSchedule(
        optimizer,
        ref_wd=wd,
        final_wd=final_wd,
        T_max=int(ipe_scale * num_epochs * iterations_per_epoch),
    )

    scaler = torch.amp.GradScaler() if mixed_precision else None
    return optimizer, scaler, scheduler, wd_scheduler


def lambda_warmup_iters(frac, total_iters):
    """[start, end] fractions of a run -> (start_iter, end_iter) for the
    context-loss warm-up.  See `optimization.lambda_warmup_frac` in train.py.
    """
    if len(frac) != 2:
        raise ValueError(f"lambda_warmup_frac must be [start, end], got {frac}")
    start_frac, end_frac = float(frac[0]), float(frac[1])
    if not 0.0 <= start_frac < end_frac <= 1.0:
        raise ValueError(
            f"lambda_warmup_frac must satisfy 0 <= start < end <= 1, got {frac}"
        )
    start_iter = int(round(start_frac * total_iters))
    end_iter = max(int(round(end_frac * total_iters)), start_iter + 1)
    return start_iter, end_iter
