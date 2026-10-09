import os
import time
import json
import logging
import random
import contextlib
import zlib
from functools import partial

import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import numpy as np
import tqdm
from absl import app, flags
from ml_collections import config_flags
from peft import PeftModel
from diffusers import WanPipeline
from diffusers.utils import export_to_video
from diffusers.models.transformers.transformer_wan import WanTransformerBlock

from flow_grpo.ood_utils import compute_text_embeddings
from flow_grpo.diffusers_patch.wan_pipeline_orig_awr import wan_pipeline_orig_awr
from flow_grpo.fsdp_utils import (
    FSDPConfig,
    fsdp_wrapper,
    init_distributed,
)


tqdm = partial(tqdm.tqdm, dynamic_ncols=True)

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/base.py", "Training configuration.")
flags.DEFINE_string("lora_path", "", "Path to LoRA checkpoint to load.")
flags.DEFINE_string("output_dir", "", "Output directory. Defaults to config.save_dir/inference.")
flags.DEFINE_integer("num_steps", 0, "Override num_inference_steps. 0 = use config.sample.eval_num_steps.")
flags.DEFINE_integer("batch_size", 0, "Override test batch size. 0 = use config.sample.test_batch_size.")
flags.DEFINE_float("guidance_scale", 0.0, "Override guidance_scale. 0 = use config.sample.guidance_scale.")
flags.DEFINE_integer("seed", None, "Base seed override. Defaults to config.seed; each prompt adds its CRC32 hash.")

logger = logging.getLogger(__name__)


class TextPromptDataset(Dataset):
    def __init__(self, dataset, split='test'):
        self.file_path = os.path.join(dataset, f'{split}.txt')
        with open(self.file_path, 'r') as f:
            self.prompts = [line.strip() for line in f.readlines()]

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx], "index": idx}

    @staticmethod
    def collate_fn(examples):
        prompts = [ex["prompt"] for ex in examples]
        indices = [ex["index"] for ex in examples]
        return prompts, indices


class GenevalPromptDataset(Dataset):
    def __init__(self, dataset, split='test'):
        self.file_path = os.path.join(dataset, f'{split}_metadata.jsonl')
        with open(self.file_path, 'r', encoding='utf-8') as f:
            self.metadatas = [json.loads(line) for line in f]
            self.prompts = [item['prompt'] for item in self.metadatas]

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx], "index": idx}

    @staticmethod
    def collate_fn(examples):
        prompts = [ex["prompt"] for ex in examples]
        indices = [ex["index"] for ex in examples]
        return prompts, indices


def get_transformer_layer_cls():
    return {WanTransformerBlock,}


def set_seed(seed, device_specific=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device_specific and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + dist.get_rank() if dist.is_initialized() else seed)


def main(_):
    config = FLAGS.config
    is_distributed, rank, world_size, local_rank = init_distributed()
    device = (
        torch.device(f'cuda:{local_rank}')
        if torch.cuda.is_available()
        else torch.device('cpu')
    )

    num_steps = FLAGS.num_steps if FLAGS.num_steps > 0 else config.sample.eval_num_steps
    test_batch_size = FLAGS.batch_size if FLAGS.batch_size > 0 else config.sample.test_batch_size
    guidance_scale = FLAGS.guidance_scale if FLAGS.guidance_scale > 0 else config.sample.guidance_scale
    base_seed = config.seed if FLAGS.seed is None else FLAGS.seed

    output_dir = FLAGS.output_dir
    if not output_dir:
        output_dir = os.path.join(config.save_dir, "inference")
    if is_distributed:
        run_id_obj = [
            time.strftime("%Y.%m.%d_%H.%M.%S", time.localtime()) if rank == 0 else None
        ]
        dist.broadcast_object_list(run_id_obj, src=0)
        unique_id = run_id_obj[0]
    else:
        unique_id = time.strftime("%Y.%m.%d_%H.%M.%S", time.localtime())
    output_dir = os.path.join(output_dir, unique_id)
    os.makedirs(output_dir, exist_ok=True)

    do_cfg = guidance_scale > 1.0

    if rank == 0:
        logger.info(f"Output: {output_dir}")
        logger.info(f"Steps: {num_steps}, Batch: {test_batch_size}, Guidance: {guidance_scale}, CFG: {do_cfg}")
        logger.info(f"Base seed: {base_seed}; per-prompt seed: base + (CRC32(prompt) & 0x7FFFFFFF)")

    inference_dtype = torch.float32
    if config.mixed_precision == "fp16":
        inference_dtype = torch.float16
    elif config.mixed_precision == "bf16":
        inference_dtype = torch.bfloat16

    pipeline = WanPipeline.from_pretrained(config.pretrained.model)

    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.transformer.requires_grad_(False)

    pipeline.set_progress_bar_config(
        position=1,
        disable=local_rank != 0,
        leave=False,
        desc="Timestep",
        dynamic_ncols=True,
    )

    pipeline.vae.to(device, dtype=torch.float32)
    pipeline.text_encoder.to(device, dtype=inference_dtype)
    text_encoders = [pipeline.text_encoder]
    tokenizers = [pipeline.tokenizer]
    transformer = pipeline.transformer.to(device)

    lora_path = FLAGS.lora_path or getattr(config.train, 'lora_path', '')
    if lora_path:
        if rank == 0:
            logger.info(f"Loading LoRA from {lora_path}")
        transformer = PeftModel.from_pretrained(transformer, lora_path)
        transformer.set_adapter("default")

    if config.use_fsdp:
        fsdp_config = FSDPConfig(
            sharding_strategy="FULL_SHARD",
            backward_prefetch="BACKWARD_PRE",
            cpu_offload=False,
            num_replicate=1,
            num_shard=world_size,
            mixed_precision_dtype=inference_dtype,
            use_activation_checkpointing=False,
            use_device_mesh=False,
        )
        transformer.to(dtype=torch.float32)
        transformer_wrapped = fsdp_wrapper(transformer, fsdp_config, get_transformer_layer_cls)
        pipeline.transformer = transformer_wrapped
    else:
        transformer.to(device=device, dtype=torch.float32)
        pipeline.transformer = transformer

    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    set_seed(base_seed, device_specific=True)

    if config.prompt_fn == "general_ocr":
        dataset_cls = TextPromptDataset
    elif config.prompt_fn == "geneval":
        dataset_cls = GenevalPromptDataset
    else:
        dataset_cls = TextPromptDataset

    test_dataset = dataset_cls(config.dataset, "test")
    test_dataloader = DataLoader(
        test_dataset,
        sampler=DistributedSampler(test_dataset, shuffle=False),
        batch_size=test_batch_size,
        collate_fn=dataset_cls.collate_fn,
        num_workers=4,
    )

    if rank == 0:
        logger.info(f"Test prompts: {len(test_dataset)}, Batches: {len(test_dataloader)}")

    pipeline.transformer.eval()
    all_video_paths = []
    all_prompts = []

    autocast_ctx = contextlib.nullcontext
    if inference_dtype == torch.bfloat16:
        autocast_ctx = lambda: torch.amp.autocast('cuda', enabled=True, dtype=torch.bfloat16)
    elif inference_dtype == torch.float16:
        autocast_ctx = lambda: torch.amp.autocast('cuda', enabled=True, dtype=torch.float16)

    neg_prompt_embed = compute_text_embeddings([""], text_encoders, tokenizers, 512, device)
    eval_neg_embeds = neg_prompt_embed.repeat(test_batch_size, 1, 1)

    for batch_idx, (prompts, indices) in tqdm(
        list(enumerate(test_dataloader)),
        desc="Inference",
        disable=local_rank != 0,
        position=0,
    ):
        prompt_embeds = compute_text_embeddings(prompts, text_encoders, tokenizers, 512, device)
        cur_neg_embeds = eval_neg_embeds[:len(prompt_embeds)]
        eval_generator = [
            torch.Generator(device=device).manual_seed(
                base_seed + (zlib.crc32(prompt.encode("utf-8")) & 0x7FFFFFFF)
            )
            for prompt in prompts
        ]
        with autocast_ctx():
            with torch.no_grad():
                videos, = wan_pipeline_orig_awr(
                    pipeline,
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=cur_neg_embeds,
                    num_inference_steps=num_steps,
                    guidance_scale=guidance_scale,
                    height=config.height,
                    width=config.width,
                    num_frames=config.frames,
                    batch_size=len(prompt_embeds),
                    device=device,
                    return_latents=False,
                    generator=eval_generator,
                )

        batch_paths = []
        for m in range(len(videos)):
            video_path = os.path.abspath(os.path.join(
                output_dir, f"video-{rank}-{batch_idx}-{m}.mp4"
            ))
            export_to_video(
                videos[m].permute(0, 2, 3, 1).float().cpu().numpy(),
                video_path,
                fps=config.fps,
            )
            batch_paths.append(video_path)
            all_prompts.append(prompts[m])

        all_video_paths.extend(batch_paths)

    gathered_paths = [None] * world_size
    dist.all_gather_object(gathered_paths, all_video_paths)
    gathered_prompts = [None] * world_size
    dist.all_gather_object(gathered_prompts, all_prompts)

    if rank == 0:
        flat_paths = [p for rp in gathered_paths for p in rp]
        flat_prompts = [p for rp in gathered_prompts for p in rp]

        manifest = []
        for path, prompt in zip(flat_paths, flat_prompts):
            manifest.append({"video_path": path, "prompt": prompt})

        manifest_path = os.path.join(output_dir, "manifest.json")
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

        logger.info(f"Generated {len(manifest)} videos -> {output_dir}")
        logger.info(f"Manifest: {manifest_path}")

    if world_size > 1:
        dist.barrier(device_ids=[local_rank])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    app.run(main)
