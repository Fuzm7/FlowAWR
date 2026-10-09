import os
import time
import json
import logging
import random
import contextlib
import tempfile
from functools import partial
from collections import defaultdict

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

from flow_grpo.rewards import multi_score
from flow_grpo.fsdp_utils import (
    FSDPConfig,
    fsdp_wrapper,
    init_distributed,
)


tqdm = partial(tqdm.tqdm, dynamic_ncols=True)

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/base.py", "Training configuration.")
flags.DEFINE_string("lora_path", "", "Path to LoRA checkpoint.")
flags.DEFINE_string("lora_path_template", "", "LoRA path template with '{}' placeholder for step, e.g. '.../checkpoint-{}-ema'. Used when lora_steps is set.")
flags.DEFINE_string("lora_steps", "", "Comma-separated checkpoint steps to evaluate, e.g. '0,40,80'. Step 0 means base model (no LoRA).")
flags.DEFINE_string("output_dir", "./evaluation_output/wan", "Output directory for results.")
flags.DEFINE_integer("num_steps", 50, "Number of inference steps.")
flags.DEFINE_integer("batch_size", 5, "Per-GPU batch size.")
flags.DEFINE_float("guidance_scale", 1.0, "Guidance scale. <=1.0 disables CFG.")
flags.DEFINE_string("reward_fn", "video_hpsv3", "Comma-separated reward function names.")
flags.DEFINE_string("method_name", "wan_baseline", "Method identifier for output files.")
flags.DEFINE_bool("save_videos", False, "Save generated videos to output_dir.")
flags.DEFINE_integer("seed", 42, "Random seed.")

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

    output_dir = FLAGS.output_dir
    os.makedirs(output_dir, exist_ok=True)

    guidance_scale = FLAGS.guidance_scale
    do_cfg = guidance_scale > 1.0
    batch_size = FLAGS.batch_size
    num_steps = FLAGS.num_steps

    if rank == 0:
        logger.info(f"Output: {output_dir}")
        logger.info(f"Steps: {num_steps}, Batch: {batch_size}, Guidance: {guidance_scale}, CFG: {do_cfg}")

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
    transformer = pipeline.transformer.to(device)

    if FLAGS.lora_steps.strip():
        eval_steps = [int(s) for s in FLAGS.lora_steps.split(",") if s.strip() != ""]
    else:
        eval_steps = None

    if eval_steps is not None and any(s != 0 for s in eval_steps) and not FLAGS.lora_path_template.strip():
        raise ValueError("lora_path_template is required when lora_steps contains non-zero steps.")

    peft_transformer = None
    adapter_name_of = {}
    if eval_steps is not None:
        for step in [s for s in eval_steps if s != 0]:
            lora_path = FLAGS.lora_path_template.format(step)
            adapter_name = f"step_{step}"
            if peft_transformer is None:
                transformer = PeftModel.from_pretrained(transformer, lora_path, adapter_name=adapter_name)
                peft_transformer = transformer
            else:
                peft_transformer.load_adapter(lora_path, adapter_name=adapter_name)
            adapter_name_of[step] = adapter_name
            if rank == 0:
                logger.info(f"Loaded LoRA adapter '{adapter_name}' from {lora_path}")
    else:
        lora_path = FLAGS.lora_path or getattr(config.train, 'lora_path', '')
        if lora_path:
            if rank == 0:
                logger.info(f"Loading LoRA from {lora_path}")
            transformer = PeftModel.from_pretrained(transformer, lora_path)
            transformer.set_adapter("default")
            peft_transformer = transformer

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
        transformer.to(device=device, dtype=inference_dtype)
        pipeline.transformer = transformer

    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    reward_names = [r.strip() for r in FLAGS.reward_fn.split(",")]
    score_dict = {name: 1.0 for name in reward_names}
    scoring_fn = multi_score(device, score_dict)

    dataset = TextPromptDataset(config.dataset, "test")
    dataloader = DataLoader(
        dataset,
        sampler=DistributedSampler(dataset, shuffle=False),
        batch_size=batch_size,
        collate_fn=TextPromptDataset.collate_fn,
        num_workers=4,
    )

    if rank == 0:
        logger.info(f"Test prompts: {len(dataset)}, Batches: {len(dataloader)}")
        logger.info(f"Reward functions: {reward_names}")

    pipeline.transformer.eval()

    autocast_ctx = contextlib.nullcontext
    if not config.use_fsdp:
        if inference_dtype == torch.bfloat16:
            autocast_ctx = lambda: torch.amp.autocast('cuda', enabled=True, dtype=torch.bfloat16)
        elif inference_dtype == torch.float16:
            autocast_ctx = lambda: torch.amp.autocast('cuda', enabled=True, dtype=torch.float16)

    def run_evaluation(method_name, adapter_name):
        set_seed(FLAGS.seed, device_specific=True)

        if adapter_name is None and peft_transformer is not None:
            adapter_ctx = peft_transformer.disable_adapter()
        elif adapter_name:
            peft_transformer.set_adapter(adapter_name)
            adapter_ctx = contextlib.nullcontext()
        else:
            adapter_ctx = contextlib.nullcontext()

        video_tmp_dir = tempfile.mkdtemp(prefix="eval_wan_")
        results_this_rank = []

        with adapter_ctx:
            for batch_idx, (prompts, indices) in tqdm(
                list(enumerate(dataloader)),
                desc=f"Eval[{method_name}]",
                disable=local_rank != 0,
                position=0,
            ):
                with autocast_ctx():
                    negative_prompt = [""] * len(prompts) if do_cfg else None
                    output = pipeline(
                        prompt=prompts,
                        negative_prompt=negative_prompt,
                        num_inference_steps=num_steps,
                        guidance_scale=guidance_scale,
                        height=config.height,
                        width=config.width,
                        num_frames=config.frames,
                        output_type="pt",
                        return_dict=False,
                    )
                    videos = output[0]

                video_paths = []
                for m in range(len(videos)):
                    video_path = os.path.abspath(os.path.join(
                        video_tmp_dir, f"eval-{rank}-{batch_idx}-{m}.mp4"
                    ))
                    export_to_video(
                        videos[m].permute(0, 2, 3, 1).float().cpu().numpy(),
                        video_path,
                        fps=config.fps,
                    )
                    video_paths.append(video_path)

                metadata = [{"prompt": p} for p in prompts]
                scores, _ = scoring_fn([videos, video_paths], prompts, metadata, only_strict=False)

                for m in range(len(prompts)):
                    result_item = {
                        "sample_id": indices[m],
                        "prompt": prompts[m],
                        "scores": {},
                    }
                    for score_name, score_values in scores.items():
                        val = score_values[m]
                        if isinstance(val, (torch.Tensor, np.ndarray)):
                            val = float(val)
                        result_item["scores"][score_name] = val

                    if FLAGS.save_videos:
                        result_item["video_path"] = video_paths[m]

                    results_this_rank.append(result_item)

                if not FLAGS.save_videos:
                    for vp in video_paths:
                        os.remove(vp)

        if not FLAGS.save_videos:
            os.rmdir(video_tmp_dir)
        elif rank == 0:
            logger.info(f"Videos saved in: {video_tmp_dir}")

        dist.barrier()

        all_gathered_results = [None] * world_size
        dist.all_gather_object(all_gathered_results, results_this_rank)

        if rank == 0:
            flat_results = [item for sublist in all_gathered_results for item in sublist]
            flat_results.sort(key=lambda x: x["sample_id"])

            results_path = os.path.join(output_dir, f"{method_name}_evaluation_results.jsonl")
            with open(results_path, "w", encoding="utf-8") as f:
                for item in flat_results:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")

            all_scores_agg = defaultdict(list)
            for result in flat_results:
                for score_name, score_value in result["scores"].items():
                    if isinstance(score_value, (int, float)):
                        all_scores_agg[score_name].append(score_value)

            average_scores = {
                name: float(np.mean([s for s in vals if s != -10.0]))
                for name, vals in all_scores_agg.items()
            }

            avg_path = os.path.join(output_dir, f"{method_name}_average_scores.json")
            with open(avg_path, "w", encoding="utf-8") as f:
                json.dump(average_scores, f, indent=2)

            logger.info(f"[{method_name}] Evaluated {len(flat_results)} samples. Results: {results_path}")
            logger.info(f"--- Average Scores [{method_name}] ---")
            for name, avg in sorted(average_scores.items()):
                logger.info(f"  {name:<20}: {avg:.4f}")

        if world_size > 1:
            dist.barrier(device_ids=[local_rank])

    if eval_steps is not None:
        for step in eval_steps:
            adapter_name = None if step == 0 else adapter_name_of[step]
            run_evaluation(f"{FLAGS.method_name}_step{step}", adapter_name)
    else:
        run_evaluation(FLAGS.method_name, "")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    app.run(main)
