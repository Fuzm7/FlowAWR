import os
import argparse
import torch
import torch.distributed as dist
from diffusers import StableDiffusion3Pipeline
from peft import PeftModel
from flow_grpo.diffusers_patch.pipeline_with_logprob import pipeline_with_logprob


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=os.environ.get("SD3_MODEL", "stabilityai/stable-diffusion-3.5-medium"))
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--prompt_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./inference_output_sd3")
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--num_steps", type=int, default=40)
    parser.add_argument("--guidance_scale", type=float, default=4.5)
    parser.add_argument("--negative_prompt", type=str, default="")
    parser.add_argument("--num_images_per_prompt", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["fp16", "bf16", "fp32"])
    return parser.parse_args()


def load_prompts(prompt_file):
    with open(prompt_file, "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f if line.strip()]
    return prompts


def main():
    args = parse_args()

    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if world_size > 1:
        dist.init_process_group("nccl")

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    os.makedirs(args.output_dir, exist_ok=True)

    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    dtype = dtype_map[args.dtype]

    pipeline = StableDiffusion3Pipeline.from_pretrained(args.model_path, torch_dtype=dtype)

    if args.lora_path:
        pipeline.transformer = PeftModel.from_pretrained(pipeline.transformer, args.lora_path)
        pipeline.transformer.merge_and_unload()

    pipeline.to(device)
    pipeline.set_progress_bar_config(disable=(rank != 0))

    all_prompts = load_prompts(args.prompt_file)
    if rank == 0:
        print(f"Total prompts: {len(all_prompts)}, GPUs: {world_size}, Images/prompt: {args.num_images_per_prompt}")

    indices = list(range(rank, len(all_prompts), world_size))
    local_prompts = [all_prompts[i] for i in indices]

    generator = torch.Generator(device=device).manual_seed(args.seed + rank)

    for local_idx in range(0, len(local_prompts), args.batch_size):
        batch_prompts = local_prompts[local_idx : local_idx + args.batch_size]
        batch_global_indices = indices[local_idx : local_idx + args.batch_size]

        with torch.no_grad():
            images, _, _ = pipeline_with_logprob(
                pipeline,
                prompt=batch_prompts,
                negative_prompt=[args.negative_prompt] * len(batch_prompts) if args.negative_prompt else None,
                num_inference_steps=args.num_steps,
                num_images_per_prompt=args.num_images_per_prompt,
                guidance_scale=args.guidance_scale,
                height=args.resolution,
                width=args.resolution,
                output_type="pil",
                noise_level=0.0,
                deterministic=True,
                solver="dpm2",
                model_type="sd3",
                generator=generator,
            )

        for i, img in enumerate(images):
            prompt_idx = batch_global_indices[i // args.num_images_per_prompt]
            img_idx = i % args.num_images_per_prompt
            save_path = os.path.join(args.output_dir, f"{prompt_idx:05d}_{img_idx}.png")
            img.save(save_path)

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()

    if rank == 0:
        total = len(all_prompts) * args.num_images_per_prompt
        print(f"Done. {total} images saved to {args.output_dir}")


if __name__ == "__main__":
    main()
