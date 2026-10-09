import os
import sys
import time
import json
import zlib
import logging
import random
import tempfile
import contextlib
from concurrent import futures
from collections import defaultdict
from functools import partial

import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import numpy as np
import wandb
import tqdm
from absl import app, flags
from ml_collections import config_flags
from peft import LoraConfig, get_peft_model, PeftModel
from diffusers import WanPipeline
from diffusers.utils import export_to_video
from diffusers.models.transformers.transformer_wan import WanTransformerBlock

import flow_grpo.rewards
from flow_grpo.stat_tracking import PerPromptStatTracker
from flow_grpo.diffusers_patch.wan_prompt_embedding import encode_prompt
from flow_grpo.diffusers_patch.wan_pipeline_orig_awr import wan_pipeline_orig_awr
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


def gather_tensor(tensor, world_size):
    if world_size == 1:
        return tensor
    gather_list = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gather_list, tensor)
    return torch.cat(gather_list)


def set_seed(seed, device_specific=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device_specific and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + dist.get_rank() if dist.is_initialized() else seed)


class TextPromptDataset(Dataset):
    def __init__(self, dataset, split='train'):
        self.file_path = os.path.join(dataset, f'{split}.txt')
        with open(self.file_path, 'r') as f:
            self.prompts = [line.strip() for line in f.readlines()]

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx], "metadata": {'prompt': self.prompts[idx]}}

    @staticmethod
    def collate_fn(examples):
        prompts = [example["prompt"] for example in examples]
        metadatas = [example["metadata"] for example in examples]
        return prompts, metadatas


class GenevalPromptDataset(Dataset):
    def __init__(self, dataset, split='train'):
        self.file_path = os.path.join(dataset, f'{split}_metadata.jsonl')
        with open(self.file_path, 'r', encoding='utf-8') as f:
            self.metadatas = [json.loads(line) for line in f]
            self.prompts = [item['prompt'] for item in self.metadatas]

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx], "metadata": self.metadatas[idx]}

    @staticmethod
    def collate_fn(examples):
        prompts = [example["prompt"] for example in examples]
        metadatas = [example["metadata"] for example in examples]
        return prompts, metadatas


class DistributedKRepeatSampler(Sampler):
    def __init__(self, dataset, batch_size, k, num_replicas, rank, seed=0):
        self.dataset = dataset
        self.batch_size = batch_size
        self.k = k
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.total_samples = self.num_replicas * self.batch_size
        assert self.total_samples % self.k == 0, (
            f"k cannot divide n*b, k{k}-num_replicas{num_replicas}-batch_size{batch_size}"
        )
        self.m = self.total_samples // self.k
        self.epoch = 0

    def __iter__(self):
        while True:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g)[:self.m].tolist()
            repeated_indices = [idx for idx in indices for _ in range(self.k)]
            shuffled_indices = torch.randperm(len(repeated_indices), generator=g).tolist()
            shuffled_samples = [repeated_indices[i] for i in shuffled_indices]
            per_card_samples = []
            for i in range(self.num_replicas):
                start = i * self.batch_size
                end = start + self.batch_size
                per_card_samples.append(shuffled_samples[start:end])
            yield per_card_samples[self.rank]

    def set_epoch(self, epoch):
        self.epoch = epoch


def compute_text_embeddings(prompt, text_encoders, tokenizers, max_sequence_length, device):
    with torch.no_grad():
        prompt_embeds = encode_prompt(
            text_encoders, tokenizers, prompt, max_sequence_length
        )
        prompt_embeds = prompt_embeds.to(device)
    return prompt_embeds


def calculate_zero_std_ratio(prompts, gathered_rewards):
    prompt_array = np.array(prompts)
    unique_prompts, inverse_indices, counts = np.unique(
        prompt_array, return_inverse=True, return_counts=True
    )
    grouped_rewards = gathered_rewards['ori_avg'][np.argsort(inverse_indices)]
    split_indices = np.cumsum(counts)[:-1]
    reward_groups = np.split(grouped_rewards, split_indices)
    prompt_std_devs = np.array([np.std(group) for group in reward_groups])
    zero_std_count = np.count_nonzero(prompt_std_devs == 0)
    zero_std_ratio = zero_std_count / len(prompt_std_devs)
    return zero_std_ratio, prompt_std_devs.mean()


def get_transformer_layer_cls():
    return {WanTransformerBlock,}


def compute_dpo_reward_local_awr(
    base_transformer, transformer_wrapped, pipeline,
    x0, initial_noise, num_train_timesteps, prompt_embeds, config,
    timestep_indices=None,
):
    device = x0.device
    B = x0.shape[0]
    v_target = (x0 - initial_noise).float()

    if timestep_indices is None:
        timestep_indices = list(range(num_train_timesteps))

    T = len(timestep_indices)
    all_scheduler_timesteps = pipeline.scheduler.timesteps
    sigmas = pipeline.scheduler.sigmas

    mse_dpo_list = []
    base_transformer.set_adapter("dpo")
    for t_idx in timestep_indices:
        sigma = sigmas[t_idx].view(1, 1, 1, 1, 1).to(device)
        hidden_states = ((1 - sigma) * x0 + sigma * initial_noise).to(prompt_embeds.dtype)
        timestep = all_scheduler_timesteps[t_idx].expand(B).to(device)
        v_dpo = transformer_wrapped(
            hidden_states=hidden_states,
            timestep=timestep,
            encoder_hidden_states=prompt_embeds,
            return_dict=False,
        )[0].float()
        mse_dpo_list.append(((v_dpo - v_target) ** 2).mean(dim=list(range(1, v_dpo.ndim))))
        del v_dpo

    mse_ref_list = []
    with base_transformer.disable_adapter():
        for t_idx in timestep_indices:
            sigma = sigmas[t_idx].view(1, 1, 1, 1, 1).to(device)
            hidden_states = ((1 - sigma) * x0 + sigma * initial_noise).to(prompt_embeds.dtype)
            timestep = all_scheduler_timesteps[t_idx].expand(B).to(device)
            v_ref = transformer_wrapped(
                hidden_states=hidden_states,
                timestep=timestep,
                encoder_hidden_states=prompt_embeds,
                return_dict=False,
            )[0].float()
            mse_ref_list.append(((v_ref - v_target) ** 2).mean(dim=list(range(1, v_ref.ndim))))
            del v_ref

    base_transformer.set_adapter("default")

    dpo_rewards = torch.zeros(B, T, device=device)
    for i in range(T):
        dpo_rewards[:, i] = -config.dpo_beta * (mse_dpo_list[i] - mse_ref_list[i])

    return dpo_rewards


def return_decay(step, decay_type):
    if decay_type == 0:
        flat, uprate, uphold = 0, 0.0, 0.0
    elif decay_type == 1:
        flat, uprate, uphold = 0, 0.001, 0.5
    elif decay_type == 2:
        flat, uprate, uphold = 75, 0.0075, 0.999
    else:
        raise ValueError(f"Unknown decay_type: {decay_type}")
    if step < flat:
        return 0.0
    return min((step - flat) * uprate, uphold)


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
                videos, = wan_pipeline_orig_awr(
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
                    return_latents=False,
                    generator=eval_generator,
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

        if getattr(config, 'use_dpo_reward', False):
            rewards = {"avg": np.zeros(len(videos))}
        else:
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
    assert not (
        getattr(config.train, "flash_mode", False)
        and getattr(config, "use_dpo_reward", False)
    ), "flash_mode 与 use_dpo_reward 不可同时启用"
    assert not (
        getattr(config.train, "ultra_flash", False)
        and not getattr(config.train, "flash_mode", False)
    ), "ultra_flash 需与 flash_mode 同时启用"
    assert not getattr(config, "ood_enable", False), (
        "本脚本不支持域外混合训练，请使用 scripts/train_wan2_1_awr_ood.py"
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
        for name, param in transformer.named_parameters():
            if "old" in name:
                param.requires_grad = False
        transformer_wrapped = DDP(
            transformer,
            device_ids=[local_rank],
            output_device=local_rank,
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

    train_sampler = DistributedKRepeatSampler(
        dataset=train_dataset,
        batch_size=config.sample.train_batch_size,
        k=config.sample.num_image_per_prompt,
        num_replicas=world_size,
        rank=rank,
        seed=42,
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

    neg_prompt_embed = compute_text_embeddings([""], text_encoders, tokenizers, 512, device)
    sample_neg_prompt_embeds = neg_prompt_embed.repeat(config.sample.train_batch_size, 1, 1)
    train_neg_prompt_embeds = neg_prompt_embed.repeat(config.train.batch_size, 1, 1)

    if config.sample.num_image_per_prompt * config.sample.sample_time_per_prompt == 1:
        config.per_prompt_stat_tracking = False
    if config.per_prompt_stat_tracking:
        stat_tracker = PerPromptStatTracker(config.sample.global_std)

    executor = futures.ThreadPoolExecutor(max_workers=8)

    first_epoch = 0
    global_step = 0

    num_train_timesteps = int(config.sample.num_steps * config.train.timestep_fraction)
    flash_mode = getattr(config.train, "flash_mode", False)
    ultra_flash = getattr(config.train, "ultra_flash", False)
    num_sample_batches = config.sample.num_batches_per_epoch * config.sample.sample_time_per_prompt
    config.train.gradient_accumulation_steps = num_sample_batches // 2 if num_sample_batches > 1 else 1
    steps_per_epoch = num_sample_batches // config.train.gradient_accumulation_steps

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

    train_iter = iter(train_dataloader)
    optimizer.zero_grad()

    for epoch in range(first_epoch, config.num_epochs):
        pipeline.transformer.eval()
        samples = []
        all_sampling_paths_local = []
        all_sampling_prompts_local = []
        eval_log = {}

        for i in tqdm(
            range(config.sample.num_batches_per_epoch),
            desc=f"Epoch {epoch}: sampling",
            disable=local_rank != 0,
            position=0,
        ):
            base_transformer.set_adapter("default")
            train_sampler.set_epoch(epoch * config.sample.num_batches_per_epoch + i)

            prompts, prompt_metadata = next(train_iter)

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

            if i == 0 and epoch % config.eval_freq == 0 and epoch >= 0:
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
            with autocast():
                with torch.no_grad():
                    videos, latents_clean, initial_noise = wan_pipeline_orig_awr(
                        pipeline,
                        prompt_embeds=prompt_embeds,
                        negative_prompt_embeds=sample_neg_prompt_embeds[:len(prompt_embeds)],
                        num_inference_steps=config.sample.num_steps,
                        guidance_scale=config.sample.guidance_scale,
                        height=config.height,
                        width=config.width,
                        num_frames=config.frames,
                        batch_size=config.sample.train_batch_size,
                        device=device,
                        return_latents=True,
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

            chosen_idx_local = None
            if flash_mode:
                gathered_prompts_lists = [None] * world_size
                dist.all_gather_object(gathered_prompts_lists, prompts)
                if ultra_flash:
                    if rank == 0:
                        flat_prompts = [p for lst in gathered_prompts_lists for p in lst]
                        chosen_flat = [
                            random.randint(0, num_train_timesteps - 1) for _ in flat_prompts
                        ]
                        container = [chosen_flat]
                    else:
                        container = [None]
                    dist.broadcast_object_list(container, src=0)
                    chosen_flat = container[0]
                    B_local = len(prompts)
                    start = rank * B_local
                    end = start + B_local
                    chosen_idx_local = torch.tensor(
                        chosen_flat[start:end], device=device, dtype=torch.long
                    )
                else:
                    if rank == 0:
                        flat_prompts = [p for lst in gathered_prompts_lists for p in lst]
                        idx_dict = {}
                        for p in flat_prompts:
                            if p not in idx_dict:
                                idx_dict[p] = random.randint(0, num_train_timesteps - 1)
                        container = [idx_dict]
                    else:
                        container = [None]
                    dist.broadcast_object_list(container, src=0)
                    idx_dict = container[0]
                    chosen_idx_local = torch.tensor(
                        [idx_dict[p] for p in prompts], device=device, dtype=torch.long
                    )

            timesteps = pipeline.scheduler.timesteps.repeat(
                config.sample.train_batch_size, 1
            )

            if getattr(config, 'use_dpo_reward', False):
                with torch.no_grad():
                    dpo_rewards = compute_dpo_reward_local_awr(
                        base_transformer, transformer_wrapped, pipeline,
                        latents_clean, initial_noise, num_train_timesteps,
                        prompt_embeds, config,
                    )
                rewards_result = dpo_rewards
            else:
                rewards_result = executor.submit(
                    reward_fn, [videos, sampling_video_path], prompts, prompt_metadata,
                    only_strict=True,
                )
            time.sleep(0)

            sample_record = {
                "prompt_ids": prompt_ids,
                "prompt_embeds": prompt_embeds,
                "timesteps": timesteps,
                "latents_clean": latents_clean,
                "rewards": rewards_result,
            }
            if flash_mode:
                sample_record["chosen_idx"] = chosen_idx_local
            samples.append(sample_record)

        if epoch < 2:
            global_step += steps_per_epoch
            continue

        for sample_item in tqdm(
            samples, desc="Waiting for rewards", disable=local_rank != 0, position=0,
        ):
            if getattr(config, 'use_dpo_reward', False):
                sample_item["rewards"] = {
                    "avg": sample_item["rewards"].to(device).float()
                }
            else:
                rewards, _ = sample_item["rewards"].result()
                sample_item["rewards"] = {
                    key: torch.as_tensor(value, device=device).float()
                    for key, value in rewards.items()
                }

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
        if not getattr(config, 'use_dpo_reward', False):
            collated_samples["rewards"]["avg"] = (
                collated_samples["rewards"]["avg"].unsqueeze(1).repeat(1, num_train_timesteps)
            )

        gathered_sampling_lists = [None] * world_size
        gathered_prompt_lists = [None] * world_size
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
            if getattr(config, 'use_dpo_reward', False):
                advantages, energies, energy_stds = stat_tracker.update(
                    prompts_all_decoded, gathered_rewards_dict["avg"], exp=True,
                )
                advantages = energies
            elif config.train.energy_mode:
                advantages, energies, energy_stds = stat_tracker.update(
                    prompts_all_decoded, gathered_rewards_dict["avg"],
                    exp=config.train.energy_mode, hard_gating=config.train.hard_gating,
                )
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
            stat_tracker.clear()
        else:
            avg_rewards_all = gathered_rewards_dict["avg"]
            advantages = (avg_rewards_all - avg_rewards_all.mean()) / (
                avg_rewards_all.std() + 1e-4
            )

        collated_samples["advantages"] = torch.from_numpy(
            advantages.reshape(world_size, -1, advantages.shape[-1])[rank]
        ).to(device)
        if config.train.energy_mode:
            collated_samples["energies"] = torch.from_numpy(
                energies.reshape(world_size, -1, energies.shape[-1])[rank]
            ).to(device)
            collated_samples["energy_stds"] = torch.from_numpy(
                energy_stds.reshape(world_size, -1, energy_stds.shape[-1])[rank]
            ).to(device)

        if rank == 0:
            logger.info(
                f"Advantages mean: {collated_samples['advantages'].abs().mean().item()}"
            )
            if config.train.energy_mode:
                logger.info(
                    f"Energies mean: {collated_samples['energies'].abs().mean().item()}"
                )

        del collated_samples["rewards"]
        del collated_samples["prompt_ids"]

        num_batches = (
            config.sample.num_batches_per_epoch * config.sample.sample_time_per_prompt
        )

        if getattr(config, 'filter_zero_advantage', False):
            filtered_key = "energies" if config.train.energy_mode else "advantages"
            mask = collated_samples[filtered_key].abs().sum(dim=1) > 1e-6
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
        effective_grad_accum_steps = (
            config.train.gradient_accumulation_steps
            * (1 if flash_mode else num_train_timesteps)
        )
        current_accumulated_steps = 0
        gradient_update_times = 0

        for inner_epoch in range(config.train.num_inner_epochs):
            perm = torch.randperm(total_batch_size_filtered, device=device)
            shuffled_filtered_samples = {k: v[perm] for k, v in filtered_samples.items()}

            training_batch_size = total_batch_size_filtered // num_batches

            samples_batched_list = []
            for k_batch in range(num_batches):
                batch_dict = {}
                start = k_batch * training_batch_size
                end = (k_batch + 1) * training_batch_size
                for key, val_tensor in shuffled_filtered_samples.items():
                    batch_dict[key] = val_tensor[start:end]
                samples_batched_list.append(batch_dict)

            info_accumulated = defaultdict(list)

            for i, train_sample_batch in tqdm(
                list(enumerate(samples_batched_list)),
                desc=f"Epoch {epoch}.{inner_epoch}: training",
                position=0,
                disable=local_rank != 0,
            ):
                if flash_mode:
                    train_timesteps = [None]
                else:
                    train_timesteps = list(range(num_train_timesteps))
                embeds = train_sample_batch["prompt_embeds"].to(inference_dtype)
                if config.train.cfg:
                    negative_embeds = train_neg_prompt_embeds[:len(embeds)].to(inference_dtype)
                else:
                    negative_embeds = None

                for j in tqdm(
                    train_timesteps,
                    desc="Timestep",
                    position=1,
                    leave=False,
                    disable=local_rank != 0,
                ):
                    x0 = train_sample_batch["latents_clean"]
                    if flash_mode:
                        idx_b = train_sample_batch["chosen_idx"]
                        t_steps = train_sample_batch["timesteps"].gather(
                            1, idx_b[:, None]
                        ).squeeze(1)
                    else:
                        idx_b = None
                        t_steps = train_sample_batch["timesteps"][:, j]
                    t = t_steps / 1000
                    t_expanded = t.view(-1, *([1] * (len(x0.shape) - 1)))

                    noise = torch.randn_like(x0.float())
                    xt = (1 - t_expanded) * x0 + t_expanded * noise
                    xt = xt.to(inference_dtype)

                    base_transformer.set_adapter("old")
                    with torch.no_grad(), train_autocast():
                        if config.train.cfg:
                            old_pred_cond = transformer_wrapped(
                                hidden_states=xt,
                                timestep=t_steps,
                                encoder_hidden_states=embeds,
                                return_dict=False,
                            )[0]
                            old_pred_uncond = transformer_wrapped(
                                hidden_states=xt,
                                timestep=t_steps,
                                encoder_hidden_states=negative_embeds,
                                return_dict=False,
                            )[0]
                            old_prediction = (
                                old_pred_uncond
                                + config.sample.guidance_scale
                                * (old_pred_cond - old_pred_uncond)
                            ).detach()
                        else:
                            old_prediction = transformer_wrapped(
                                hidden_states=xt,
                                timestep=t_steps,
                                encoder_hidden_states=embeds,
                                return_dict=False,
                            )[0].detach()

                    base_transformer.set_adapter("default")

                    if config.train.beta > 0:
                        with torch.no_grad():
                            if config.use_lora:
                                with base_transformer.disable_adapter(), train_autocast():
                                    ref_forward_prediction = transformer_wrapped(
                                        hidden_states=xt,
                                        timestep=t_steps,
                                        encoder_hidden_states=embeds,
                                        return_dict=False,
                                    )[0].detach()
                            else:
                                raise ValueError(
                                    "Non-LoRA mode not supported for Wan2.1 reference model"
                                )

                    base_transformer.set_adapter("default")
                    with train_autocast():
                        forward_prediction = transformer_wrapped(
                            hidden_states=xt,
                            timestep=t_steps,
                            encoder_hidden_states=embeds,
                            return_dict=False,
                        )[0]

                    loss_terms = {}

                    if not config.train.energy_mode:
                        if flash_mode:
                            cur_advantages = train_sample_batch["advantages"].gather(
                                1, idx_b[:, None]
                            ).squeeze(1)
                        else:
                            cur_advantages = train_sample_batch["advantages"][:, j]
                        advantages_clip = torch.clamp(
                            cur_advantages,
                            -config.train.adv_clip_max,
                            config.train.adv_clip_max,
                        )
                        if hasattr(config.train, "adv_mode"):
                            if config.train.adv_mode == "positive_only":
                                advantages_clip = torch.clamp(
                                    advantages_clip, 0, config.train.adv_clip_max
                                )
                            elif config.train.adv_mode == "negative_only":
                                advantages_clip = torch.clamp(
                                    advantages_clip, -config.train.adv_clip_max, 0
                                )
                            elif config.train.adv_mode == "one_only":
                                advantages_clip = torch.where(
                                    advantages_clip > 0,
                                    torch.ones_like(advantages_clip),
                                    torch.zeros_like(advantages_clip),
                                )
                            elif config.train.adv_mode == "binary":
                                advantages_clip = torch.sign(advantages_clip)
                    else:
                        if flash_mode:
                            cur_energies = train_sample_batch["energies"].gather(
                                1, idx_b[:, None]
                            ).squeeze(1)
                            cur_energy_stds = train_sample_batch["energy_stds"].gather(
                                1, idx_b[:, None]
                            ).squeeze(1)
                        else:
                            cur_energies = train_sample_batch["energies"][:, j]
                            cur_energy_stds = train_sample_batch["energy_stds"][:, j]

                    loss_terms["x0_norm"] = torch.mean(x0**2).detach()
                    loss_terms["x0_norm_max"] = torch.max(x0**2).detach()
                    loss_terms["old_deviate"] = torch.mean(
                        (forward_prediction - old_prediction) ** 2
                    ).detach()
                    loss_terms["old_deviate_max"] = torch.max(
                        (forward_prediction - old_prediction) ** 2
                    ).detach()

                    if not config.train.energy_mode:
                        positive_prediction = (
                            config.beta * forward_prediction
                            + (1 - config.beta) * old_prediction.detach()
                        )
                        implicit_negative_prediction = (
                            (1.0 + config.beta) * old_prediction.detach()
                            - config.beta * forward_prediction
                        )
                        normalized_advantages_clip = (
                            advantages_clip / config.train.adv_clip_max
                        ) / 2.0 + 0.5
                        r = torch.clamp(normalized_advantages_clip, 0, 1)

                        x0_prediction = xt - t_expanded * positive_prediction
                        with torch.no_grad():
                            weight_factor = (
                                torch.abs(x0_prediction.double() - x0.double())
                                .mean(dim=tuple(range(1, x0.ndim)), keepdim=True)
                                .clip(min=0.00001)
                            )
                        positive_loss = (
                            (x0_prediction - x0) ** 2 / weight_factor
                        ).mean(dim=tuple(range(1, x0.ndim)))

                        negative_x0_prediction = (
                            xt - t_expanded * implicit_negative_prediction
                        )
                        with torch.no_grad():
                            negative_weight_factor = (
                                torch.abs(
                                    negative_x0_prediction.double() - x0.double()
                                )
                                .mean(dim=tuple(range(1, x0.ndim)), keepdim=True)
                                .clip(min=0.00001)
                            )
                        negative_loss = (
                            (negative_x0_prediction - x0) ** 2
                            / negative_weight_factor
                        ).mean(dim=tuple(range(1, x0.ndim)))

                        ori_policy_loss = (
                            r * positive_loss / config.beta
                            + (1.0 - r) * negative_loss / config.beta
                        )
                        policy_loss = (
                            ori_policy_loss * config.train.adv_clip_max
                        ).mean()
                    else:
                        energies_expanded = cur_energies.view(
                            -1, *([1] * (len(x0.shape) - 1))
                        )
                        pred_x0_old = xt - t_expanded * old_prediction.detach()
                        target_x0 = pred_x0_old + energies_expanded * (
                            x0 - pred_x0_old
                        )

                        x0_prediction = xt - t_expanded * forward_prediction
                        with torch.no_grad():
                            weight_factor = (
                                torch.abs(x0_prediction.double() - x0.double())
                                .mean(dim=tuple(range(1, x0.ndim)), keepdim=True)
                                .clip(min=0.00001)
                            )
                        ori_policy_loss = (
                            (x0_prediction - target_x0) ** 2 / weight_factor
                        ).mean(dim=tuple(range(1, x0.ndim)))
                        policy_loss = ori_policy_loss.mean()

                        loss_terms["min_ori_e_std"] = cur_energy_stds.min().detach()
                        loss_terms["energy_mean"] = cur_energies.mean().detach()
                        loss_terms["energy_std"] = cur_energies.std(correction=0).detach()
                        loss_terms["energy_max"] = cur_energies.max().detach()
                        loss_terms["energy_min"] = cur_energies.min().detach()
                        loss_terms["energy_pos_rate"] = (
                            (cur_energies > 0).float().mean().detach()
                        )
                        loss_terms["energy_clip_rate"] = (
                            (torch.abs(cur_energies) >= 2.99).float().mean().detach()
                        )

                    loss = policy_loss
                    loss_terms["policy_loss"] = policy_loss.detach()
                    loss_terms["unweighted_policy_loss"] = (
                        ori_policy_loss.mean().detach()
                    )

                    if config.train.beta > 0:
                        kl_div_loss = (
                            (forward_prediction - ref_forward_prediction) ** 2
                        ).mean(dim=tuple(range(1, x0.ndim)))

                        loss += config.train.beta * torch.mean(kl_div_loss)
                        kl_div_loss = torch.mean(kl_div_loss)
                        loss_terms["kl_div_loss"] = kl_div_loss.detach()
                        loss_terms["kl_div"] = torch.mean(
                            (
                                (forward_prediction - ref_forward_prediction) ** 2
                            ).mean(dim=tuple(range(1, x0.ndim)))
                        ).detach()
                        loss_terms["old_kl_div"] = torch.mean(
                            (
                                (old_prediction - ref_forward_prediction) ** 2
                            ).mean(dim=tuple(range(1, x0.ndim)))
                        ).detach()

                    loss_terms["total_loss"] = loss.detach()

                    scaled_loss = loss / effective_grad_accum_steps
                    is_last_accum = (current_accumulated_steps + 1) % effective_grad_accum_steps == 0
                    sync_ctx = contextlib.nullcontext if is_last_accum else transformer_wrapped.no_sync
                    with sync_ctx():
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

                        log_info = {
                            k: torch.mean(torch.stack(v_list))
                            for k, v_list in info_accumulated.items()
                        }
                        for k, v in log_info.items():
                            dist.all_reduce(v, op=dist.ReduceOp.SUM)
                            log_info[k] = (v / world_size).item()
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
