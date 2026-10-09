import os
import time
import zlib
import logging
import random
import contextlib
from concurrent import futures
from collections import defaultdict
from functools import partial

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
import wandb
import tqdm
from absl import app, flags
from ml_collections import config_flags
from peft import LoraConfig, get_peft_model, PeftModel
from diffusers import WanPipeline
from diffusers.utils import export_to_video

import flow_grpo.rewards
from flow_grpo.stat_tracking import PerPromptStatTracker
from flow_grpo.ood_utils import (
    calculate_zero_std_ratio,
    compute_ood_group_stats,
    compute_text_embeddings,
    gather_tensor,
    get_transformer_layer_cls,
    load_ood_latents,
    return_decay,
    set_seed,
    DistributedKRepeatSampler,
    GenevalPromptDataset,
    OODPromptDataset,
    TextPromptDataset,
    UnifiedOODSampler,
)
from flow_grpo.diffusers_patch.wan_pipeline_flash_grpo import (
    compute_coe_table,
    wan_pipeline_flash_grpo,
)
from flow_grpo.fsdp_utils import (
    FSDPConfig,
    fsdp_wrapper,
    init_distributed,
    save_fsdp_checkpoint,
    register_optimizer_offload_hooks,
    sync_lora_adapters_per_block,
)
from flow_grpo.ema import EMAModuleWrapper


tqdm = partial(tqdm.tqdm, dynamic_ncols=True)

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/base.py", "Training configuration.")

logger = logging.getLogger(__name__)

TRAIN_LOG_METRIC_KEYS = (
    "policy_loss",
    "approx_kl",
    "clipfrac",
    "ratio_mean",
    "kl_loss",
    "sft_loss",
    "total_loss",
    "ood_sft_weight",
)


def get_flash_value_tensor(timesteps, coe_table):
    return torch.tensor(
        [coe_table[int(timestep)] for timestep in timesteps.tolist()],
        device=timesteps.device,
        dtype=torch.float32,
    )


def combine_flash_grpo_losses(
    policy_losses, sft_losses, batch_size, ood_sft_weight, kl_losses=None,
    beta=0.0,
):
    total = policy_losses.sum()
    if kl_losses is not None:
        total = total + beta * kl_losses.sum()
    if sft_losses is not None:
        total = total + ood_sft_weight * sft_losses.sum()
    return total / batch_size


def reduce_sum_count(sums, counts, device):
    stats = torch.stack([
        torch.stack(sums).to(device=device, dtype=torch.float32),
        torch.as_tensor(counts, device=device, dtype=torch.float32),
    ])
    if dist.is_initialized():
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    return stats


def compute_flash_grpo_loss(
    transformer_wrapped, base_transformer, pipeline, batch, embeds, negative_embeds,
    config, coe_table, value_norm, train_autocast, ood_sft_weight=1.0,
):
    """计算固定前传日程下的逐样本 PPO、KL 和 OOD SFT 损失。"""
    device = batch["latents"].device
    batch_size = batch["latents"].shape[0]
    chosen_idx = batch["chosen_idx"]
    rollout_before = batch["latents"][:, 0]
    rollout_after = batch["latents"][:, 1]
    rollout_timestep = batch["timesteps"].gather(
        1, chosen_idx[:, None]
    ).squeeze(1)
    is_ood = batch.get(
        "is_ood", torch.zeros(batch_size, device=device, dtype=torch.bool)
    )
    rollout_mask = ~is_ood

    model_input = rollout_before
    model_timestep = rollout_timestep
    ood_noise = None
    if bool(is_ood.any()):
        ood_x0 = rollout_before[is_ood].float()
        ood_timestep = rollout_timestep[is_ood]
        ood_t = ood_timestep / 1000
        ood_t_expanded = ood_t.view(
            -1, *([1] * (rollout_before.ndim - 1))
        )
        ood_noise = torch.randn_like(ood_x0)
        ood_xt = (1 - ood_t_expanded) * ood_x0 + ood_t_expanded * ood_noise

        model_input = rollout_before.clone()
        model_input[is_ood] = ood_xt.to(model_input.dtype)
        model_timestep = rollout_timestep.clone()
        model_timestep[is_ood] = ood_timestep

    with train_autocast():
        conditional_pred = transformer_wrapped(
            hidden_states=model_input,
            timestep=model_timestep,
            encoder_hidden_states=embeds,
            return_dict=False,
        )[0]
        if config.train.cfg:
            unconditional_pred = transformer_wrapped(
                hidden_states=model_input,
                timestep=model_timestep,
                encoder_hidden_states=negative_embeds,
                return_dict=False,
            )[0]
        else:
            unconditional_pred = None

    ref_noise_pred = None
    if config.train.beta > 0:
        ref_noise_pred = compute_ref_for_kl(
            base_transformer,
            transformer_wrapped,
            model_input,
            model_timestep,
            embeds,
            negative_embeds,
            config,
            train_autocast,
        )

    policy_losses = torch.empty(0, device=device, dtype=torch.float32)
    kl_losses = None
    loss_terms = {}
    if bool(rollout_mask.any()):
        if config.train.cfg:
            noise_pred = (
                unconditional_pred[rollout_mask]
                + config.sample.guidance_scale
                * (
                    conditional_pred[rollout_mask]
                    - unconditional_pred[rollout_mask]
                )
            )
        else:
            noise_pred = conditional_pred[rollout_mask]

        from flow_grpo.diffusers_patch.wan_pipeline_with_logprob import (
            sde_step_with_logprob,
        )
        _, log_prob, prev_sample_mean, _, _ = (
            sde_step_with_logprob(
                pipeline.scheduler,
                noise_pred.float(),
                rollout_timestep[rollout_mask],
                rollout_before[rollout_mask].float(),
                prev_sample=rollout_after[rollout_mask].float(),
                return_dt_and_std_dev_t=True,
            )
        )

        stored_log_prob = batch["log_probs"][rollout_mask, 0]
        ratio = torch.exp(log_prob - stored_log_prob)
        advantages = batch["advantages"].gather(
            1, chosen_idx[:, None]
        ).squeeze(1)[rollout_mask]
        advantages_clip = torch.clamp(
            advantages,
            -config.train.adv_clip_max,
            config.train.adv_clip_max,
        )
        values = get_flash_value_tensor(
            rollout_timestep[rollout_mask], coe_table
        )
        value_weights = values / value_norm
        unclipped_loss = -value_weights * advantages_clip * ratio
        clipped_loss = -value_weights * advantages_clip * torch.clamp(
            ratio,
            1.0 - config.train.clip_range,
            1.0 + config.train.clip_range,
        )
        policy_losses = torch.maximum(unclipped_loss, clipped_loss)

        loss_terms["policy_loss"] = policy_losses.mean().detach()
        loss_terms["approx_kl"] = (
            0.5 * torch.mean((log_prob - stored_log_prob) ** 2)
        ).detach()
        loss_terms["clipfrac"] = torch.mean(
            (torch.abs(ratio - 1.0) > config.train.clip_range).float()
        ).detach()
        loss_terms["ratio_mean"] = ratio.mean().detach()

        if ref_noise_pred is not None:
            _, _, ref_prev_sample_mean, ref_std_dev_t, ref_sqrt_dt = (
                sde_step_with_logprob(
                    pipeline.scheduler,
                    ref_noise_pred[rollout_mask].float(),
                    rollout_timestep[rollout_mask],
                    rollout_before[rollout_mask].float(),
                    prev_sample=rollout_after[rollout_mask].float(),
                    return_dt_and_std_dev_t=True,
                )
            )
            kl_losses = compute_flash_kl_losses(
                prev_sample_mean,
                ref_prev_sample_mean,
                ref_std_dev_t,
                ref_sqrt_dt,
            )
            loss_terms["kl_loss"] = kl_losses.mean().detach()

    sft_losses = None
    if bool(is_ood.any()):
        sft_losses, sft_terms = compute_ood_sft_loss(
            conditional_pred[is_ood],
            rollout_before[is_ood],
            ood_noise,
        )
        loss_terms.update(sft_terms)

    loss = combine_flash_grpo_losses(
        policy_losses,
        sft_losses,
        batch_size,
        ood_sft_weight,
        kl_losses=kl_losses,
        beta=config.train.beta,
    )
    if unconditional_pred is not None:
        loss = loss + unconditional_pred.float().reshape(-1)[0] * 0.0
    loss_terms["total_loss"] = loss.detach()
    return loss, loss_terms


def compute_ref_for_kl(
    base_transformer, transformer_wrapped, model_input, timestep,
    embeds, negative_embeds, config, train_autocast,
):
    """用完整 batch 的固定前传日程计算 reference velocity。"""
    with (
        torch.no_grad(),
        base_transformer.disable_adapter(),
        train_autocast(),
    ):
        ref_pred_text = transformer_wrapped(
            hidden_states=model_input,
            timestep=timestep,
            encoder_hidden_states=embeds,
            return_dict=False,
        )[0]
        if config.train.cfg:
            ref_pred_uncond = transformer_wrapped(
                hidden_states=model_input,
                timestep=timestep,
                encoder_hidden_states=negative_embeds,
                return_dict=False,
            )[0]
            return (
                ref_pred_uncond
                + config.sample.guidance_scale
                * (ref_pred_text - ref_pred_uncond)
            )
        return ref_pred_text


def compute_ood_sft_loss(v_pred, x0, noise):
    """返回 OOD offline 样本的逐样本 flow-matching SFT 损失。"""
    v_target = (noise - x0).float()
    per_sample_loss = ((v_pred.float() - v_target) ** 2).mean(
        dim=tuple(range(1, v_pred.ndim))
    )
    return per_sample_loss, {"sft_loss": per_sample_loss.mean().detach()}


def compute_flash_kl_losses(
    prev_sample_mean, ref_prev_sample_mean, std_dev_t, sqrt_dt,
):
    squared_mean_delta = (
        (prev_sample_mean - ref_prev_sample_mean) ** 2
    ).mean(dim=tuple(range(1, prev_sample_mean.ndim)))
    transition_variance = (
        std_dev_t * sqrt_dt
    ).flatten(start_dim=1)[:, 0] ** 2
    return squared_mean_delta / (2 * transition_variance)


def compute_global_value_norms(
    samples_batched_list, gradient_accumulation_steps, coe_table, device,
):
    """计算每个 optimizer step 在全部 rank 上共享的 rollout value 均值。"""
    assert len(samples_batched_list) % gradient_accumulation_steps == 0, (
        f"训练批次数 {len(samples_batched_list)} 不能被梯度累积步数 "
        f"{gradient_accumulation_steps} 整除"
    )
    value_norms = []
    for group_start in range(
        0, len(samples_batched_list), gradient_accumulation_steps
    ):
        local_sum = torch.zeros((), device=device, dtype=torch.float32)
        local_count = torch.zeros((), device=device, dtype=torch.float32)
        for batch in samples_batched_list[
            group_start:group_start + gradient_accumulation_steps
        ]:
            batch_size = batch["timesteps"].shape[0]
            is_ood = batch.get(
                "is_ood",
                torch.zeros(batch_size, device=device, dtype=torch.bool),
            )
            rollout_mask = ~is_ood
            if not bool(rollout_mask.any()):
                continue
            chosen_idx = batch["chosen_idx"][rollout_mask]
            timesteps = batch["timesteps"][rollout_mask].gather(
                1, chosen_idx[:, None]
            ).squeeze(1)
            values = get_flash_value_tensor(timesteps, coe_table)
            local_sum += values.sum()
            local_count += values.numel()

        value_stats = reduce_sum_count(
            [local_sum], [local_count.item()], device
        ).flatten()
        assert value_stats[1].item() > 0, (
            f"optimizer step {len(value_norms)} 不含 rollout 样本"
        )
        value_norms.append(value_stats[0] / value_stats[1])
    return value_norms


def eval_fn(
    pipeline, test_dataloader, text_encoders, tokenizers, config,
    rank, local_rank, world_size, device, global_step,
    reward_fn, executor, autocast, ema, transformer_trainable_parameters, epoch,
):
    if config.train.ema and ema is not None:
        ema.copy_ema_to(transformer_trainable_parameters, store_temp=True)

    neg_prompt_embed = compute_text_embeddings([""], text_encoders, tokenizers, 512, device)
    eval_neg_embeds = neg_prompt_embed.repeat(config.sample.test_batch_size, 1, 1)

    all_eval_items = []
    for batch_idx, test_batch in enumerate(tqdm(
        test_dataloader, desc="Eval: ", disable=local_rank != 0, position=0,
    )):
        prompts, prompt_metadata = test_batch
        prompt_embeds = compute_text_embeddings(prompts, text_encoders, tokenizers, 512, device)
        cur_neg_embeds = eval_neg_embeds[:len(prompt_embeds)]

        eval_generator = [
            torch.Generator(device=device).manual_seed(
                config.seed + (zlib.crc32(prompt.encode("utf-8")) & 0x7FFFFFFF)
            )
            for prompt in prompts
        ]
        with autocast():
            with torch.no_grad():
                videos, latents, log_probs, _ = wan_pipeline_flash_grpo(
                    pipeline,
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=cur_neg_embeds,
                    num_inference_steps=config.sample.eval_num_steps,
                    guidance_scale=config.sample.guidance_scale,
                    height=config.height,
                    width=config.width,
                    num_frames=config.frames,
                    batch_size=len(prompt_embeds),
                    device=device,
                    index=config.sample.eval_num_steps // 2,
                    generator=None,
                )

        save_root = os.path.join(config.save_dir, "eval", f"eval_{global_step}")
        os.makedirs(save_root, exist_ok=True)

        path = []
        for m in range(len(videos)):
            video_path = os.path.abspath(os.path.join(save_root, f"eval-{rank}-{batch_idx}-{m}.mp4"))
            export_to_video(
                videos[m].permute(0, 2, 3, 1).float().cpu().numpy(), video_path, fps=config.fps
            )
            path.append(video_path)

        rewards = executor.submit(
            reward_fn, [videos, path], prompts, prompt_metadata, only_strict=False
        )
        time.sleep(0)
        rewards, _ = rewards.result()

        for m in range(len(path)):
            item_rewards = {k: float(v[m]) for k, v in rewards.items()}
            all_eval_items.append((path[m], prompts[m], item_rewards))

    gathered_eval_items = [None] * world_size
    dist.all_gather_object(gathered_eval_items, all_eval_items)
    all_items_flat = [item for lst in gathered_eval_items for item in lst]

    dedup = {}
    for eval_item in all_items_flat:
        dedup.setdefault(eval_item[1], eval_item)
    all_items_flat = [dedup[prompt] for prompt in sorted(dedup)]
    reward_keys = all_items_flat[0][2].keys() if all_items_flat else []
    all_rewards = {
        key: np.array([item[2][key] for item in all_items_flat])
        for key in reward_keys
    }

    eval_log = {}
    if rank == 0:
        num_samples = min(48, len(all_items_flat))
        eval_log = {
            "eval_images": [
                wandb.Video(
                    all_items_flat[idx][0],
                    caption=f"{all_items_flat[idx][1]:.1000} | " + " | ".join(
                        f"{k}: {v:.2f}" for k, v in all_items_flat[idx][2].items() if v != -10
                    ),
                    format="mp4",
                )
                for idx in range(num_samples)
            ],
            **{
                f"eval_reward_{key}": np.mean(value[value != -10])
                for key, value in all_rewards.items()
            },
        }

    if config.train.ema and ema is not None:
        ema.copy_temp_to(transformer_trainable_parameters)

    if world_size > 1:
        dist.barrier(device_ids=[local_rank])

    return eval_log


def main(_):
    config = FLAGS.config
    is_distributed, rank, world_size, local_rank = init_distributed()
    assert not getattr(config, "ood_enable", False) or len(config.reward_fn) == 1, (
        f"ood_enable 要求 reward_fn 恰含 1 个通道（域外分数为单一标量），"
        f"当前为 {sorted(config.reward_fn)}"
    )
    ood_reward_weight = (
        next(iter(config.reward_fn.values()))
        if getattr(config, "ood_enable", False) and config.reward_fn
        else 1.0
    )
    device = (
        torch.device(f'cuda:{local_rank}')
        if torch.cuda.is_available()
        else torch.device('cpu')
    )

    if is_distributed:
        run_id_obj = [
            time.strftime("%Y.%m.%d_%H.%M.%S", time.localtime()) if rank == 0 else None
        ]
        dist.broadcast_object_list(run_id_obj, src=0)
        unique_id = run_id_obj[0]
    else:
        unique_id = time.strftime("%Y.%m.%d_%H.%M.%S", time.localtime())

    if not config.run_name:
        config.run_name = unique_id

    config.save_freq = config.eval_freq
    config.save_dir = os.path.join(config.save_dir, config.run_name, unique_id)
    os.makedirs(config.save_dir, exist_ok=True)

    if rank == 0:
        wandb.init(
            project="Video-AWR",
            name=config.run_name + "_" + unique_id,
            config=config.to_dict(),
            dir=config.save_dir,
        )
    logger.info(f"\n{config}")
    set_seed(config.seed, device_specific=True)

    inference_dtype = torch.float32
    if config.mixed_precision == "fp16":
        inference_dtype = torch.float16
        autocast = lambda: torch.amp.autocast('cuda', enabled=True, dtype=torch.float16)
    elif config.mixed_precision == "bf16":
        inference_dtype = torch.bfloat16
        autocast = lambda: torch.amp.autocast('cuda', enabled=True, dtype=torch.bfloat16)
    else:
        autocast = contextlib.nullcontext

    if config.use_fsdp:
        train_autocast = contextlib.nullcontext
    else:
        train_autocast = autocast

    pipeline = WanPipeline.from_pretrained(config.pretrained.model)

    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.transformer.requires_grad_(not config.use_lora)

    text_encoders = [pipeline.text_encoder]
    tokenizers = [pipeline.tokenizer]

    pipeline.set_progress_bar_config(
        position=1,
        disable=local_rank != 0,
        leave=False,
        desc="Timestep",
        dynamic_ncols=True,
    )

    pipeline.vae.to(device, dtype=torch.float32)
    pipeline.text_encoder.to(device, dtype=inference_dtype)
    transformer = pipeline.transformer.to(device)

    if config.use_lora:
        target_modules = [
            "to_k", "to_q", "to_v", "to_out.0",
            "add_k_proj", "add_q_proj", "add_v_proj", "to_add_out",
        ]
        transformer_lora_config = LoraConfig(
            r=32,
            lora_alpha=64,
            init_lora_weights="gaussian",
            target_modules=target_modules,
        )

        if config.train.lora_path:
            transformer = PeftModel.from_pretrained(transformer, config.train.lora_path)
            transformer.set_adapter("default")
        else:
            transformer = get_peft_model(transformer, transformer_lora_config)

        transformer.add_adapter("old", transformer_lora_config)
        for name, param in transformer.named_parameters():
            if "default" in name or "old" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        transformer.set_adapter("default")

    if getattr(config, 'dpo_lora_path', ''):
        transformer.load_adapter(config.dpo_lora_path, adapter_name="dpo")
        transformer.set_adapter("default")

    for name, param in transformer.named_parameters():
        if "old" in name:
            param.requires_grad = False

    base_transformer = transformer

    if config.use_fsdp:
        fsdp_config = FSDPConfig(
            sharding_strategy="FULL_SHARD",
            backward_prefetch="BACKWARD_PRE",
            cpu_offload=False,
            num_replicate=1,
            num_shard=world_size,
            mixed_precision_dtype=inference_dtype,
            use_activation_checkpointing=True,
            use_device_mesh=False,
        )
        transformer.to(dtype=torch.float32)
        transformer_wrapped = fsdp_wrapper(transformer, fsdp_config, get_transformer_layer_cls)
    else:
        transformer.to(device=device, dtype=torch.float32)
        transformer.enable_gradient_checkpointing()
        transformer_wrapped = DDP(
            transformer,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            find_unused_parameters=False,
        )
    if config.use_fsdp:
        pipeline.transformer = transformer_wrapped
    else:
        pipeline.transformer = base_transformer

    params_dict = dict(transformer_wrapped.named_parameters())
    transformer_trainable_parameters = []
    old_transformer_trainable_parameters = []
    for name, param in params_dict.items():
        if "default" in name and param.requires_grad:
            old_name = name.replace("default", "old")
            if old_name in params_dict:
                transformer_trainable_parameters.append(param)
                old_transformer_trainable_parameters.append(params_dict[old_name])
            else:
                raise ValueError(f"Parameter mapping error: {old_name} not found for {name}")

    ema = None
    if config.train.ema:
        ema = EMAModuleWrapper(
            transformer_trainable_parameters, decay=0.9, update_step_interval=1, device=device
        )

    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    optimizer = torch.optim.AdamW(
        transformer_trainable_parameters,
        lr=config.train.learning_rate,
        betas=(config.train.adam_beta1, config.train.adam_beta2),
        weight_decay=config.train.adam_weight_decay,
        eps=config.train.adam_epsilon,
    )
    if config.use_fsdp:
        register_optimizer_offload_hooks(optimizer)

    if config.prompt_fn == "general_ocr":
        train_dataset_cls, test_dataset_cls = TextPromptDataset, TextPromptDataset
    elif config.prompt_fn == "geneval":
        train_dataset_cls, test_dataset_cls = GenevalPromptDataset, GenevalPromptDataset
    else:
        raise NotImplementedError(f"Unsupported prompt_fn: {config.prompt_fn}")

    train_dataset = train_dataset_cls(config.dataset, "train")
    test_dataset = test_dataset_cls(config.dataset, "test")

    ood_enable = getattr(config, "ood_enable", False)
    n_ood_per_prompt = getattr(config.sample, "n_ood_per_prompt", 0)
    ood_dataset = None
    if ood_enable:
        assert getattr(config.sample, "ood_group_mode", "replace") == "replace", (
            "统一采样器只实现 replace 模式，"
            f"得到 ood_group_mode={config.sample.ood_group_mode!r}"
        )
        k_group = config.sample.num_image_per_prompt
        num_unique_prompts = config.sample.unique_prompts
        assert num_unique_prompts > 0, (
            "OOD 训练需通过 config.sample.unique_prompts 指定每 epoch 组数"
        )
        ood_prompts_epoch = int(
            round(config.sample.ood_ratio * num_unique_prompts)
        )
        assert ood_prompts_epoch >= 1, (
            f"ood_ratio={config.sample.ood_ratio} 过小: 每 epoch 域外 prompt "
            f"数为 0 (unique_prompts={num_unique_prompts})"
        )
        m_ood = ood_prompts_epoch
        m_pure = num_unique_prompts - ood_prompts_epoch
        ood_dataset = OODPromptDataset(config.ood_json, n_ood_per_prompt)

        # 统一 sampler 的唯一整除约束是
        # (unique_prompts*k) % (world_size*num_batches) == 0，与 ood_ratio、
        # n_ood 无关。num_batches 依赖 train_batch_size 推导，故实例在下方
        # 派生块构造，此处只做与 num_batches 无关的前置校验。
        assert m_pure >= 1, (
            f"ood_ratio={config.sample.ood_ratio} 过大: 域内 prompt 数={m_pure}，"
            f"须至少保留 1 个域内 prompt"
        )
        assert n_ood_per_prompt < k_group, (
            f"replace 模式 n_ood 必须 < k，得到 n_ood={n_ood_per_prompt} k={k_group}"
        )
        assert m_ood >= 1, (
            f"ood_ratio={config.sample.ood_ratio} 过小: 域外 prompt 数为 {m_ood}"
        )

    if ood_enable:
        # 统一路径不走 DataLoader，域内与域外 prompt 均由 UnifiedOODSampler 直接产出。
        train_sampler = None
        train_dataloader = None
    else:
        train_sampler = DistributedKRepeatSampler(
            dataset=train_dataset,
            batch_size=config.sample.train_batch_size,
            k=config.sample.num_image_per_prompt,
            num_replicas=world_size,
            rank=rank,
            seed=config.seed,
        )
        train_dataloader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            num_workers=1,
            collate_fn=train_dataset_cls.collate_fn,
        )
    test_dataloader = DataLoader(
        test_dataset,
        sampler=DistributedSampler(test_dataset, shuffle=False, drop_last=False),
        batch_size=config.sample.test_batch_size,
        collate_fn=test_dataset_cls.collate_fn,
        num_workers=8,
    )

    # 训练侧批次划分与采样侧解耦：训练把收集到的全部样本按每卡 train_batch_size
    # 均分。采样侧的 num_batches_per_epoch 只决定采样循环次数与推理负载。
    unified_ood_sampler = None
    if ood_enable:
        # 统一 sampler：每卡每 epoch 的样本总量恒为 unique_prompts*k // world_size
        # （与 ood_ratio/n_ood 无关），故训练切分回退到非 OOD 公式。
        # num_batches_per_epoch 由本式唯一决定并覆写 config。
        total_slots = config.sample.unique_prompts * config.sample.num_image_per_prompt
        assert total_slots % (world_size * config.sample.train_batch_size) == 0, (
            f"replace 模式 slot 池不可分割: total={total_slots}, "
            f"world_size{world_size}*train_batch_size"
            f"{config.sample.train_batch_size}={world_size * config.sample.train_batch_size}"
        )
        config.sample.num_batches_per_epoch = (
            total_slots // (world_size * config.sample.train_batch_size)
        )
        assert config.sample.num_batches_per_epoch >= 1, (
            f"推导 num_batches_per_epoch 失败: total={total_slots} "
            f"world_size={world_size} train_batch_size={config.sample.train_batch_size}"
        )
        unified_ood_sampler = UnifiedOODSampler(
            pure_dataset=train_dataset,
            ood_dataset=ood_dataset,
            m_epoch=config.sample.unique_prompts,
            k=config.sample.num_image_per_prompt,
            n_ood=n_ood_per_prompt,
            m_pure=m_pure,
            num_replicas=world_size,
            rank=rank,
            num_batches=config.sample.num_batches_per_epoch,
            seed=config.seed,
        )
        per_rank_clean_total = unified_ood_sampler.per_rank_clean_total
        logger.info(
            f"OOD enabled [replace, unified]: "
            f"unique_prompts={config.sample.unique_prompts} "
            f"m_pure={m_pure} m_ood={m_ood} n_ood={n_ood_per_prompt} | "
            f"per_rank_clean_total={per_rank_clean_total} "
            f"batch_size={unified_ood_sampler.batch_size} "
            f"batches={config.sample.num_batches_per_epoch} | "
            f"pure_pool={len(train_dataset)} ood_pool={len(ood_dataset)}"
        )
    else:
        per_rank_clean_total = (
            config.sample.train_batch_size * config.sample.num_batches_per_epoch
        )

    neg_prompt_embed = compute_text_embeddings([""], text_encoders, tokenizers, 512, device)
    sample_neg_prompt_embeds = neg_prompt_embed.repeat(config.sample.train_batch_size, 1, 1)

    if config.sample.num_image_per_prompt * config.sample.sample_time_per_prompt == 1:
        config.per_prompt_stat_tracking = False
    if config.per_prompt_stat_tracking:
        stat_tracker = PerPromptStatTracker(config.sample.global_std)

    executor = futures.ThreadPoolExecutor(max_workers=8)

    first_epoch = 0
    global_step = 0

    num_train_timesteps = int(config.sample.num_steps * config.train.timestep_fraction)
    num_train_batches = per_rank_clean_total // config.sample.train_batch_size
    assert num_train_batches >= 1, (
        f"训练批次划分失败: per_rank_clean_total={per_rank_clean_total}"
        f" train_batch_size={config.sample.train_batch_size}"
    )
    assert not ood_enable or num_train_batches % 2 == 0, (
        f"zone 划分要求训练批次数为偶数，得到 num_train_batches={num_train_batches}"
        f" (per_rank_clean_total={per_rank_clean_total},"
        f" train_batch_size={config.sample.train_batch_size})"
    )
    config.train.gradient_accumulation_steps = (
        num_train_batches // 2 if num_train_batches > 1 else 1
    )
    steps_per_epoch = num_train_batches // config.train.gradient_accumulation_steps

    samples_per_epoch = config.sample.train_batch_size * world_size * config.sample.num_batches_per_epoch
    total_train_batch_size = (
        config.train.batch_size * world_size * config.train.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num Epochs = {config.num_epochs}")
    logger.info(f"  Sample batch size per device = {config.sample.train_batch_size}")
    logger.info(f"  Train batch size per device = {config.train.batch_size}")
    logger.info(f"  Gradient Accumulation steps = {config.train.gradient_accumulation_steps}")
    logger.info(f"  Total number of samples per epoch = {samples_per_epoch}")
    logger.info(f"  Total train batch size = {total_train_batch_size}")
    logger.info(
        f"  Number of gradient updates per inner epoch = {samples_per_epoch // total_train_batch_size}"
    )
    logger.info(f"  Number of inner epochs = {config.train.num_inner_epochs}")

    reward_fn = getattr(flow_grpo.rewards, 'multi_score')(device, config.reward_fn)

    if config.use_fsdp:
        sync_lora_adapters_per_block(transformer_wrapped, decay=0.0)
    else:
        with torch.no_grad():
            for p_def, p_old in zip(transformer_trainable_parameters, old_transformer_trainable_parameters):
                p_old.data.copy_(p_def.data)

    train_iter = iter(train_dataloader) if train_dataloader is not None else None
    ood_iter = iter(unified_ood_sampler) if ood_enable else None
    optimizer.zero_grad()

    coe_table = compute_coe_table(pipeline.scheduler, config.sample.num_steps)

    for epoch in range(first_epoch, config.num_epochs):
        pipeline.transformer.eval()
        samples = []
        all_sampling_paths_local = []
        all_sampling_prompts_local = []
        all_ood_items_local = []
        eval_log = {}
        idx_dict = {}

        for i in tqdm(
            range(config.sample.num_batches_per_epoch),
            desc=f"Epoch {epoch}: sampling",
            disable=local_rank != 0,
            position=0,
        ):
            base_transformer.set_adapter("default")

            ood_rollout_prompts = []
            ood_rollout_metadata = []
            ood_offline_prompts = []
            ood_offline_metadata = []
            ood_video_paths = []
            ood_offline_rewards = []

            if ood_enable:
                # 统一 sampler 直接产出本卡本批次的 rollout/offline 槽位。
                unified_ood_sampler.set_epoch(
                    epoch * config.sample.num_batches_per_epoch + i
                )
                rollout_slots, offline_slots = next(ood_iter)
                prompts = []
                prompt_metadata = []
                for kind, idx, replica_id in rollout_slots:
                    if kind == UnifiedOODSampler.KIND_PURE:
                        item = train_dataset[idx]
                    else:
                        item = ood_dataset[idx]
                    prompts.append(item["prompt"])
                    prompt_metadata.append(item["metadata"])
                    ood_rollout_prompts.append(item["prompt"])
                    ood_rollout_metadata.append(item["metadata"])
                for kind, idx, replica_id in offline_slots:
                    item = ood_dataset[idx]
                    entry = item["ood_entries"][replica_id]
                    prompts.append(item["prompt"])
                    prompt_metadata.append(item["metadata"])
                    ood_offline_prompts.append(item["prompt"])
                    ood_offline_metadata.append(item["metadata"])
                    ood_video_paths.append(entry["video_path"])
                    ood_offline_rewards.append(float(entry["reward"]))
            else:
                train_sampler.set_epoch(
                    epoch * config.sample.num_batches_per_epoch + i
                )
                prompts, prompt_metadata = next(train_iter)

            num_rollout = len(prompts) - len(ood_offline_prompts)
            rollout_prompts = prompts[:num_rollout]
            rollout_metadata = prompt_metadata[:num_rollout]

            prompt_embeds = compute_text_embeddings(
                prompts, text_encoders, tokenizers, 512, device
            )
            prompt_ids = pipeline.tokenizer(
                prompts,
                padding="max_length",
                max_length=512,
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(device)

            if (
                i == 0
                and epoch % config.eval_freq == 0
                and not (config.debug and epoch == 0)
            ):
                eval_log = eval_fn(
                    pipeline, test_dataloader, text_encoders, tokenizers, config,
                    rank, local_rank, world_size, device, global_step,
                    reward_fn, executor, autocast, ema, transformer_trainable_parameters,
                    epoch,
                )
                if rank == 0 and eval_log:
                    wandb.log(eval_log, step=global_step)

            if i == 0 and epoch % config.save_freq == 0 and epoch > 0:
                if config.use_fsdp:
                    save_fsdp_checkpoint(
                        config.save_dir, transformer_wrapped, global_step, rank,
                        base_model=base_transformer,
                    )
                else:
                    if rank == 0:
                        save_path = os.path.join(config.save_dir, "checkpoints", f"checkpoint-{global_step}")
                        os.makedirs(save_path, exist_ok=True)
                        base_transformer.save_pretrained(save_path)
                    dist.barrier()
                if config.train.ema:
                    ema.copy_ema_to(transformer_trainable_parameters, store_temp=True)
                    if config.use_fsdp:
                        save_fsdp_checkpoint(
                            config.save_dir, transformer_wrapped, global_step, rank, ema=True,
                            base_model=base_transformer,
                        )
                    else:
                        if rank == 0:
                            save_path = os.path.join(config.save_dir, "checkpoints", f"checkpoint-{global_step}-ema")
                            os.makedirs(save_path, exist_ok=True)
                            base_transformer.save_pretrained(save_path)
                        dist.barrier()
                    ema.copy_temp_to(transformer_trainable_parameters)

            if epoch < 2:
                continue

            base_transformer.set_adapter("old")
            sampling_video_path = []
            # 同一 epoch 的 online/offline 按 prompt 共享注入点，跨批次复用。
            gathered_prompts_lists = [None] * world_size
            dist.all_gather_object(gathered_prompts_lists, prompts)
            if rank == 0:
                flat_prompts = [p for lst in gathered_prompts_lists for p in lst]
                for p in flat_prompts:
                    if p not in idx_dict:
                        idx_dict[p] = random.randint(0, num_train_timesteps - 1)
                container = [idx_dict]
            else:
                container = [None]
            dist.broadcast_object_list(container, src=0)
            idx_dict = container[0]
            chosen_idx_batch = torch.tensor(
                [idx_dict[p] for p in prompts], device=device, dtype=torch.long
            )
            chosen_idx = chosen_idx_batch[:num_rollout]

            with autocast():
                with torch.no_grad():
                    videos, latents_pair, log_probs, _ = wan_pipeline_flash_grpo(
                        pipeline,
                        prompt_embeds=prompt_embeds[:num_rollout],
                        negative_prompt_embeds=sample_neg_prompt_embeds[:num_rollout],
                        num_inference_steps=config.sample.num_steps,
                        guidance_scale=config.sample.guidance_scale,
                        height=config.height,
                        width=config.width,
                        num_frames=config.frames,
                        batch_size=num_rollout,
                        device=device,
                        index=chosen_idx,
                        generator=None,
                    )

            sampling_export_root = os.path.join(
                config.save_dir, "sampling", f"sampling_{epoch}"
            )
            os.makedirs(sampling_export_root, exist_ok=True)
            for m in range(len(videos)):
                sampling_export_path = os.path.abspath(os.path.join(
                    sampling_export_root, f"sampling-{rank}-{i}-{m}.mp4"
                ))
                export_to_video(
                    videos[m].permute(0, 2, 3, 1).float().cpu().numpy(),
                    sampling_export_path,
                    fps=config.fps,
                )
                sampling_video_path.append(sampling_export_path)
                all_sampling_paths_local.append(sampling_export_path)
                all_sampling_prompts_local.append(prompts[m])

            base_transformer.set_adapter("default")

            # latents_pair = [before, after]，log_probs = [log_prob_at_index]
            latents_before = torch.stack([latents_pair[0]], dim=1)
            latents_after = torch.stack([latents_pair[1]], dim=1)
            log_probs_t = torch.stack([log_probs[0]], dim=1)
            latents_combined = torch.cat([latents_before, latents_after], dim=1)
            log_probs_combined = log_probs_t

            rewards_result = executor.submit(
                reward_fn, [videos, sampling_video_path], rollout_prompts,
                rollout_metadata, only_strict=True,
            )
            time.sleep(0)

            if ood_enable:
                ood_latents = load_ood_latents(
                    pipeline, ood_video_paths, config.height, config.width,
                    config.frames, device,
                ).to(latents_before.dtype)
                for offline_index, video_path in enumerate(ood_video_paths):
                    all_ood_items_local.append((
                        video_path,
                        ood_offline_prompts[offline_index],
                        ood_offline_rewards[offline_index],
                    ))

            timesteps = pipeline.scheduler.timesteps.repeat(len(prompts), 1)[:, :num_train_timesteps]

            if ood_enable:
                num_ood = len(ood_video_paths)
                ood_latents_rep = torch.stack([ood_latents, ood_latents], dim=1)
                ood_log_probs = torch.zeros(
                    (num_ood, 1), device=device, dtype=torch.float32
                )
                latents_combined = torch.cat([latents_combined, ood_latents_rep], dim=0)
                log_probs_combined = torch.cat([log_probs_combined, ood_log_probs], dim=0)

            sample_record = {
                "prompt_ids": prompt_ids,
                "prompt_embeds": prompt_embeds,
                "timesteps": timesteps,
                "latents": latents_combined,
                "log_probs": log_probs_combined,
                "chosen_idx": chosen_idx_batch,
                "rewards": rewards_result,
            }
            if ood_enable:
                sample_record["is_ood"] = torch.cat([
                    torch.zeros(num_rollout, device=device, dtype=torch.bool),
                    torch.ones(len(ood_video_paths), device=device, dtype=torch.bool),
                ])
                sample_record["ood_offline_rewards"] = torch.cat([
                    torch.full(
                        (num_rollout,), float("nan"), device=device, dtype=torch.float32
                    ),
                    torch.tensor(
                        ood_offline_rewards, device=device, dtype=torch.float32
                    ),
                ])
            samples.append(sample_record)

        if epoch < 2:
            global_step += steps_per_epoch
            continue

        for sample_item in tqdm(
            samples, desc="Waiting for rewards", disable=local_rank != 0, position=0,
        ):
            rewards, _ = sample_item["rewards"].result()
            rewards = {
                key: torch.as_tensor(value, device=device).float()
                for key, value in rewards.items()
            }
            if ood_enable:
                offline = sample_item["ood_offline_rewards"]
                offline = offline[~torch.isnan(offline)]
                rewards = {
                    key: torch.cat([value, offline * ood_reward_weight])
                    for key, value in rewards.items()
                }
            sample_item["rewards"] = rewards

        collated_samples = {}
        for k in samples[0].keys():
            if isinstance(samples[0][k], dict):
                merged_dict = {}
                for sk in samples[0][k].keys():
                    valid_tensors = [s[k][sk] for s in samples if s[k][sk] is not None]
                    if valid_tensors:
                        merged_dict[sk] = torch.cat(valid_tensors, dim=0)
                if merged_dict:
                    collated_samples[k] = merged_dict
            else:
                valid_tensors = [s[k] for s in samples if s[k] is not None]
                if valid_tensors:
                    collated_samples[k] = torch.cat(valid_tensors, dim=0)

        collated_samples["rewards"]["ori_avg"] = collated_samples["rewards"]["avg"]
        collated_samples["rewards"]["avg"] = (
            collated_samples["rewards"]["avg"].unsqueeze(1).repeat(1, num_train_timesteps)
        )

        gathered_sampling_lists = [None] * world_size
        gathered_prompt_lists = [None] * world_size
        if ood_enable and all_ood_items_local:
            local_ood_mask = collated_samples["is_ood"]
            local_ood_scores = (
                collated_samples["rewards"]["ori_avg"][local_ood_mask].cpu().numpy()
            )
            for (video_path, ood_prompt, offline_reward), rescore in zip(
                all_ood_items_local, local_ood_scores
            ):
                all_sampling_paths_local.append(video_path)
                all_sampling_prompts_local.append(
                    f"[OOD] rescore={rescore:.3f} offline={offline_reward:.3f} | {ood_prompt}"
                )
        dist.all_gather_object(gathered_sampling_lists, all_sampling_paths_local)
        dist.all_gather_object(gathered_prompt_lists, all_sampling_prompts_local)
        all_sampling_paths = []
        all_sampling_prompts = []
        for paths, prts in zip(gathered_sampling_lists, gathered_prompt_lists):
            all_sampling_paths.extend(paths)
            all_sampling_prompts.extend(prts)

        gathered_rewards_dict = {}
        for key, value_tensor in collated_samples["rewards"].items():
            gathered_rewards_dict[key] = gather_tensor(value_tensor, world_size).cpu().numpy()

        gathered_ood_mask = None
        if ood_enable:
            gathered_ood_mask = gather_tensor(
                collated_samples["is_ood"].float(), world_size
            ).cpu().numpy().astype(bool)

        wandb_video_count = getattr(config.sample, 'wandb_video_count', 16)

        if rank == 0:
            epoch_log = {
                "epoch": epoch,
                **{
                    f"reward_{key}": value.mean()
                    for key, value in gathered_rewards_dict.items()
                    if '_strict_accuracy' not in key and '_accuracy' not in key
                },
                **eval_log,
            }
            if all_sampling_paths and epoch % 10 == 0:
                display_paths = all_sampling_paths[-wandb_video_count:]
                display_prompts = all_sampling_prompts[-wandb_video_count:]
                epoch_log["sample_videos"] = [
                    wandb.Video(
                        vp, caption=pr[:120], format="mp4"
                    )
                    for vp, pr in zip(display_paths, display_prompts)
                ]
            wandb.log(epoch_log, step=global_step)

        if config.per_prompt_stat_tracking:
            prompt_ids_all = gather_tensor(
                collated_samples["prompt_ids"], world_size
            ).cpu().numpy()
            prompts_all_decoded = pipeline.tokenizer.batch_decode(
                prompt_ids_all, skip_special_tokens=True
            )
            if ood_enable:
                # Offline 样本仅做 SFT，不参与在线 advantage 的均值和标准差。
                keep = ~gathered_ood_mask
                sub_advantages = stat_tracker.update(
                    [p for p, keep_flag in zip(prompts_all_decoded, keep) if keep_flag],
                    gathered_rewards_dict["avg"][keep],
                )
                advantages = np.zeros_like(
                    gathered_rewards_dict["avg"], dtype=sub_advantages.dtype
                )
                advantages[keep] = sub_advantages
            else:
                advantages = stat_tracker.update(
                    prompts_all_decoded, gathered_rewards_dict["avg"]
                )

            if rank == 0:
                group_size, trained_prompt_num = stat_tracker.get_stats()
                zero_std_ratio, reward_std_mean = calculate_zero_std_ratio(
                    prompts_all_decoded, gathered_rewards_dict
                )
                wandb.log(
                    {
                        "group_size": group_size,
                        "trained_prompt_num": trained_prompt_num,
                        "zero_std_ratio": zero_std_ratio,
                        "reward_std_mean": reward_std_mean,
                        "mean_reward_100": stat_tracker.get_mean_of_top_rewards(100),
                        "mean_reward_75": stat_tracker.get_mean_of_top_rewards(75),
                        "mean_reward_50": stat_tracker.get_mean_of_top_rewards(50),
                        "mean_reward_25": stat_tracker.get_mean_of_top_rewards(25),
                        "mean_reward_10": stat_tracker.get_mean_of_top_rewards(10),
                    },
                    step=global_step,
                )
                if ood_enable and gathered_ood_mask is not None:
                    # 域外样本在组内相对 on-policy 样本的胜率与领先幅度，
                    # 口径与 AWR OOD 脚本一致，供两条路径对照。
                    reward_flat = gathered_rewards_dict["ori_avg"]
                    ood_log = {
                        "ood_reward_mean": float(reward_flat[gathered_ood_mask].mean()),
                        "rollout_reward_mean": float(
                            reward_flat[~gathered_ood_mask].mean()
                        ),
                    }
                    ood_log.update(
                        compute_ood_group_stats(
                            prompts_all_decoded, reward_flat, gathered_ood_mask
                        )
                    )
                    wandb.log(ood_log, step=global_step)
            stat_tracker.clear()
        else:
            avg_rewards_all = gathered_rewards_dict["avg"]
            if ood_enable:
                keep = ~gathered_ood_mask
                online_rewards = avg_rewards_all[keep]
                advantages = np.zeros_like(avg_rewards_all)
                advantages[keep] = (online_rewards - online_rewards.mean()) / (
                    online_rewards.std() + 1e-4
                )
            else:
                advantages = (avg_rewards_all - avg_rewards_all.mean()) / (
                    avg_rewards_all.std() + 1e-4
                )

        collated_samples["advantages"] = torch.from_numpy(
            advantages.reshape(world_size, -1, advantages.shape[-1])[rank]
        ).to(device)

        if rank == 0:
            logger.info(
                f"Advantages mean: {collated_samples['advantages'].abs().mean().item()}"
            )

        del collated_samples["rewards"]
        del collated_samples["prompt_ids"]
        if ood_enable:
            del collated_samples["ood_offline_rewards"]

        # 训练切分用训练侧批次划分，与采样循环的 num_batches_per_epoch 解耦。
        num_batches = num_train_batches

        if getattr(config, 'filter_zero_advantage', False):
            filtered_key = "advantages"
            mask = collated_samples[filtered_key].abs().sum(dim=1) > 1e-6
            if getattr(config, "ood_enable", False):
                mask = mask | collated_samples["is_ood"]
            true_count = mask.sum().item()
            if true_count == 0:
                logger.warning(
                    f"All samples have zero {filtered_key} at epoch {epoch}. Adding noise."
                )
                collated_samples[filtered_key] = collated_samples[filtered_key] + 1e-6
                mask = collated_samples[filtered_key].abs().sum(dim=1) > 1e-6

            if true_count % num_batches != 0:
                false_indices = torch.where(~mask)[0]
                num_to_change = num_batches - (true_count % num_batches)
                if len(false_indices) >= num_to_change:
                    random_indices = torch.randperm(len(false_indices))[:num_to_change]
                    mask[false_indices[random_indices]] = True

            if rank == 0:
                wandb.log(
                    {
                        "actual_batch_size": mask.sum().item()
                        // config.sample.num_batches_per_epoch,
                    },
                    step=global_step,
                )

            filtered_samples = {k: v[mask] for k, v in collated_samples.items()}
        else:
            filtered_samples = collated_samples
        total_batch_size_filtered = filtered_samples["timesteps"].shape[0]

        pipeline.transformer.train()
        effective_grad_accum_steps = config.train.gradient_accumulation_steps
        assert num_batches % effective_grad_accum_steps == 0, (
            f"训练批次数 {num_batches} 不能被梯度累积步数 "
            f"{effective_grad_accum_steps} 整除"
        )
        current_accumulated_steps = 0
        gradient_update_times = 0

        for inner_epoch in range(config.train.num_inner_epochs):
            # zone 内洗牌：采样侧把 slot 池对半切成两个 zone 并按块号顺序铺开，
            # 拼接后前一半样本属 zone A、后一半属 zone B。全局洗牌会跨 zone 混样，
            # 使单个 optimizer step 拿到的域外 prompt 数偏离 m_ood/2，故只在半区内重排。
            # 样本数为奇数时无法对半（filter_zero_advantage 可能造成），退回全局洗牌。
            if ood_enable and total_batch_size_filtered % 2 == 0:
                zone_size = total_batch_size_filtered // 2
                perm = torch.cat([
                    torch.randperm(zone_size, device=device),
                    torch.randperm(zone_size, device=device) + zone_size,
                ])
            else:
                perm = torch.randperm(total_batch_size_filtered, device=device)
            shuffled_filtered_samples = {k: v[perm] for k, v in filtered_samples.items()}

            base_size, remainder = divmod(total_batch_size_filtered, num_batches)

            samples_batched_list = []
            start = 0
            for k_batch in range(num_batches):
                batch_dict = {}
                end = start + base_size + (1 if k_batch < remainder else 0)
                for key, val_tensor in shuffled_filtered_samples.items():
                    batch_dict[key] = val_tensor[start:end]
                samples_batched_list.append(batch_dict)
                start = end

            value_norms = compute_global_value_norms(
                samples_batched_list,
                effective_grad_accum_steps,
                coe_table,
                device,
            )
            info_accumulated = defaultdict(list)

            for i, train_sample_batch in tqdm(
                list(enumerate(samples_batched_list)),
                desc=f"Epoch {epoch}.{inner_epoch}: training",
                position=0,
                disable=local_rank != 0,
            ):
                embeds = train_sample_batch["prompt_embeds"].to(inference_dtype)
                if config.train.cfg:
                    negative_embeds = neg_prompt_embed.repeat(
                        len(embeds), 1, 1
                    ).to(inference_dtype)
                else:
                    negative_embeds = None

                ood_sft_weight = getattr(config.train, "ood_sft_weight", 1.0)
                is_last_accum = (
                    (current_accumulated_steps + 1)
                    % effective_grad_accum_steps
                    == 0
                )
                sync_ctx = (
                    contextlib.nullcontext
                    if is_last_accum
                    else transformer_wrapped.no_sync
                )

                with sync_ctx():
                    value_norm = value_norms[
                        i // effective_grad_accum_steps
                    ]
                    loss, loss_terms = compute_flash_grpo_loss(
                        transformer_wrapped,
                        base_transformer,
                        pipeline,
                        train_sample_batch,
                        embeds,
                        negative_embeds,
                        config,
                        coe_table,
                        value_norm,
                        train_autocast,
                        ood_sft_weight=ood_sft_weight,
                    )
                    loss_terms["ood_sft_weight"] = torch.tensor(
                        ood_sft_weight, device=device
                    ).detach()

                    scaled_loss = loss / effective_grad_accum_steps
                    scaled_loss.backward()
                current_accumulated_steps += 1

                for k_info, v_info in loss_terms.items():
                    info_accumulated[k_info].append(v_info)

                if current_accumulated_steps % effective_grad_accum_steps == 0:
                    if config.use_fsdp:
                        transformer_wrapped.clip_grad_norm_(config.train.max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(
                            transformer_trainable_parameters, config.train.max_grad_norm
                        )
                    optimizer.step()
                    gradient_update_times += 1
                    optimizer.zero_grad()

                    metric_sums = []
                    metric_counts = []
                    for metric_key in TRAIN_LOG_METRIC_KEYS:
                        metric_values = info_accumulated.get(metric_key, ())
                        if metric_values:
                            stacked_values = torch.stack(metric_values).float()
                            metric_sums.append(stacked_values.sum())
                            metric_counts.append(float(stacked_values.numel()))
                        else:
                            metric_sums.append(torch.zeros((), device=device))
                            metric_counts.append(0.0)
                    metric_stats = reduce_sum_count(
                        metric_sums, metric_counts, device
                    )
                    log_info = {
                        metric_key: (
                            metric_stats[0, metric_index]
                            / metric_stats[1, metric_index]
                        ).item()
                        for metric_index, metric_key in enumerate(
                            TRAIN_LOG_METRIC_KEYS
                        )
                        if metric_stats[1, metric_index].item() > 0
                    }
                    log_info.update(
                        {"epoch": epoch, "inner_epoch": inner_epoch}
                    )
                    if rank == 0:
                        wandb.log(
                            {
                                "step": global_step,
                                "gradient_update_times": gradient_update_times,
                                "epoch": epoch,
                                "inner_epoch": inner_epoch,
                                **log_info,
                            },
                            step=global_step,
                        )

                    global_step += 1
                    info_accumulated = defaultdict(list)

                if (
                    config.train.ema
                    and ema is not None
                    and (current_accumulated_steps % effective_grad_accum_steps == 0)
                ):
                    ema.step(transformer_trainable_parameters, global_step)

        if world_size > 1:
            dist.barrier(device_ids=[local_rank])

        sync_decay = return_decay(global_step, config.decay_type)
        if config.use_fsdp:
            sync_lora_adapters_per_block(transformer_wrapped, decay=sync_decay)
        else:
            with torch.no_grad():
                for p_def, p_old in zip(transformer_trainable_parameters, old_transformer_trainable_parameters):
                    p_old.data.copy_(sync_decay * p_old.data + (1 - sync_decay) * p_def.data)

    if rank == 0:
        wandb.finish()


if __name__ == "__main__":
    app.run(main)
