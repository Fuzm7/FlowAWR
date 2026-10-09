import os
import argparse
import json
import torch
import torch.distributed as dist
from diffusers import FluxPipeline
from peft import PeftModel
from flow_grpo.diffusers_patch.pipeline_with_logprob import pipeline_with_logprob
from flow_grpo.diffusers_patch.train_dreambooth_lora_flux import encode_prompt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=os.environ.get("FLUX_MODEL", "black-forest-labs/FLUX.1-dev"))
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--prompt_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./inference_output")
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--max_sequence_length", type=int, default=128)
    parser.add_argument("--guidance_scale", type=float, default=3.5)
    parser.add_argument("--num_images_per_prompt", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["fp16", "bf16", "fp32"])
    return parser.parse_args()


def load_prompts(prompt_file):
    with open(prompt_file, "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f if line.strip()]
    return prompts


def load_pipeline(args, device, dtype):
    pipeline = FluxPipeline.from_pretrained(args.model_path)
    if args.lora_path:
        pipeline.transformer = PeftModel.from_pretrained(
            pipeline.transformer, args.lora_path, adapter_name="default"
        )
        pipeline.transformer.set_adapter("default")

    pipeline.vae.to(device, dtype=torch.float32)
    pipeline.text_encoder.to(device, dtype=dtype)
    pipeline.text_encoder_2.to(device, dtype=dtype)
    pipeline.transformer.to(device, dtype=dtype)
    for model in (pipeline.vae, pipeline.text_encoder, pipeline.text_encoder_2, pipeline.transformer):
        model.requires_grad_(False)
        model.eval()
    return pipeline


@torch.no_grad()
def generate_batch(pipeline, prompts, args, device, dtype, generator):
    prompt_embeds, pooled_prompt_embeds, _ = encode_prompt(
        text_encoders=[pipeline.text_encoder, pipeline.text_encoder_2],
        tokenizers=[pipeline.tokenizer, pipeline.tokenizer_2],
        prompt=prompts,
        max_sequence_length=args.max_sequence_length,
        device=device,
        num_images_per_prompt=args.num_images_per_prompt,
    )
    with torch.autocast("cuda", enabled=(dtype != torch.float32), dtype=dtype):
        images, _, _, _, _ = pipeline_with_logprob(
            pipeline,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            num_inference_steps=args.num_steps,
            num_images_per_prompt=1,
            max_sequence_length=args.max_sequence_length,
            guidance_scale=args.guidance_scale,
            height=args.resolution,
            width=args.resolution,
            output_type="pil",
            noise_level=0.0,
            deterministic=True,
            solver="dpm2",
            model_type="flux",
            generator=generator,
        )
    return images


def main():
    args = parse_args()

    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if world_size > 1:
        dist.init_process_group("nccl")

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    os.makedirs(args.output_dir, exist_ok=True)
    if rank == 0:
        with open(os.path.join(args.output_dir, "run_config.json"), "w", encoding="utf-8") as f:
            json.dump({**vars(args), "world_size": world_size, "solver": "dpm2", "allow_tf32": True}, f, indent=2)

    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    dtype = dtype_map[args.dtype]

    pipeline = load_pipeline(args, device, dtype)
    pipeline.set_progress_bar_config(disable=(rank != 0))

    all_prompts = load_prompts(args.prompt_file)
    if rank == 0:
        print(f"Total prompts: {len(all_prompts)}, GPUs: {world_size}, Images/prompt: {args.num_images_per_prompt}")

    indices = list(range(rank, len(all_prompts), world_size))
    local_prompts = [all_prompts[i] for i in indices]

    generator = torch.Generator(device=device).manual_seed(args.seed + rank)
    manifest_path = os.path.join(args.output_dir, f"manifest_rank{rank:02d}.jsonl")
    with open(manifest_path, "w", encoding="utf-8") as manifest:
        for local_idx in range(0, len(local_prompts), args.batch_size):
            batch_prompts = local_prompts[local_idx : local_idx + args.batch_size]
            batch_global_indices = indices[local_idx : local_idx + args.batch_size]
            images = generate_batch(pipeline, batch_prompts, args, device, dtype, generator)

            for i, img in enumerate(images):
                prompt_idx = batch_global_indices[i // args.num_images_per_prompt]
                img_idx = i % args.num_images_per_prompt
                image_path = f"{prompt_idx:05d}_{img_idx}.png"
                img.save(os.path.join(args.output_dir, image_path))
                manifest.write(json.dumps({
                    "prompt_file": args.prompt_file,
                    "prompt_index": prompt_idx,
                    "image_index": img_idx,
                    "prompt": all_prompts[prompt_idx],
                    "image_path": image_path,
                }, ensure_ascii=False) + "\n")
            manifest.flush()

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()

    if rank == 0:
        total = len(all_prompts) * args.num_images_per_prompt
        print(f"Done. {total} images saved to {args.output_dir}")


if __name__ == "__main__":
    main()
