import argparse
import os
import json
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

from diffusers import StableDiffusion3Pipeline
from torch.utils.data import DataLoader, Dataset
from peft import LoraConfig, get_peft_model

from flow_grpo.rewards import multi_score

import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from collections import defaultdict
from peft import PeftModel

import logging

logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)


def setup_distributed(rank, world_size):
    """Initializes the distributed process group."""
    os.environ["MASTER_ADDR"] = os.getenv("MASTER_ADDR", "localhost")
    os.environ["MASTER_PORT"] = os.getenv("MASTER_PORT", "12355")
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup_distributed():
    """Destroys the distributed process group."""
    dist.destroy_process_group()


def is_main_process(rank):
    """Checks if the current process is the main one (rank 0)."""
    return rank == 0


class TextPromptDataset(Dataset):
    def __init__(self, dataset_path, split="test"):
        self.file_path = os.path.join(dataset_path, f"{split}.txt")
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Dataset file not found at {self.file_path}")
        with open(self.file_path, "r", encoding="utf-8") as f:
            self.prompts = [line.strip() for line in f.readlines()]

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx], "metadata": {}, "original_index": idx}


class GenevalPromptDataset(Dataset):
    def __init__(self, dataset_path, split="test"):
        self.file_path = os.path.join(dataset_path, f"{split}_metadata.jsonl")
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Dataset file not found at {self.file_path}")
        with open(self.file_path, "r", encoding="utf-8") as f:
            self.metadatas = [json.loads(line) for line in f]
            self.prompts = [item["prompt"] for item in self.metadatas]

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx], "metadata": self.metadatas[idx], "original_index": idx}


def collate_fn(examples):
    prompts = [example["prompt"] for example in examples]
    metadatas = [example["metadata"] for example in examples]
    indices = [example["original_index"] for example in examples]
    return prompts, metadatas, indices


def main(args):
    # --- Distributed Setup ---
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    setup_distributed(rank, world_size)
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    # --- Mixed Precision Setup ---
    mixed_precision_dtype = None
    if args.mixed_precision == "fp16":
        mixed_precision_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        mixed_precision_dtype = torch.bfloat16

    enable_amp = mixed_precision_dtype is not None
    if is_main_process(rank):
        print(f"Running distributed evaluation with {world_size} GPUs.")
        if enable_amp:
            print(f"Using mixed precision: {args.mixed_precision}")
    
    # --- Load Model and Pipeline ---
    if is_main_process(rank):
        print("Loading model and pipeline...")

    if args.model_type == "sd3":
        pipeline = StableDiffusion3Pipeline.from_pretrained(args.pretrained_model)
        target_modules = [
            "attn.add_k_proj",
            "attn.add_q_proj",
            "attn.add_v_proj",
            "attn.to_add_out",
            "attn.to_k",
            "attn.to_out.0",
            "attn.to_q",
            "attn.to_v",
        ]
        transformer_lora_config = LoraConfig(
            r=32, lora_alpha=64, init_lora_weights="gaussian", target_modules=target_modules
        )
    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if args.lora_hf_path:
        pipeline.transformer = PeftModel.from_pretrained(pipeline.transformer, args.lora_hf_path)
        pipeline.transformer = pipeline.transformer.merge_and_unload()
    elif args.checkpoint_path:
        lora_path = os.path.join(args.checkpoint_path, "lora")
        if is_main_process(rank):
            print(f"Loading LoRA weights from: {lora_path}")
        if not os.path.exists(lora_path):
            raise FileNotFoundError(
                f"LoRA directory not found at {lora_path}. Ensure your checkpoint has a 'lora' subdirectory."
            )

        pipeline.transformer = get_peft_model(pipeline.transformer, transformer_lora_config)
        pipeline.transformer.load_adapter(lora_path, adapter_name="default", is_trainable=False)

    pipeline.transformer.eval()
    text_encoder_dtype = mixed_precision_dtype if enable_amp else torch.float32

    pipeline.transformer.to(device, dtype=text_encoder_dtype)
    pipeline.vae.to(device, dtype=torch.float32)  # VAE usually fp32
    pipeline.text_encoder.to(device, dtype=text_encoder_dtype)
    pipeline.text_encoder_2.to(device, dtype=text_encoder_dtype)
    pipeline.text_encoder_3.to(device, dtype=text_encoder_dtype)

    pipeline.safety_checker = None
    pipeline.set_progress_bar_config(
        position=1,
        disable=not is_main_process(rank),
        leave=False,
        desc="Timestep",
        dynamic_ncols=True,
    )

    # [Start of Loop]
    for cur_dataset in args.dataset:
        base_output_dir = os.path.join(args.output_dir, cur_dataset)

        if is_main_process(rank):
            print(f"\n{'='*40}")
            print(f"Starting Evaluation for Dataset: {cur_dataset}")
            print(f"{'='*40}\n")
            os.makedirs(base_output_dir, exist_ok=True)
            if args.save_images:
                os.makedirs(os.path.join(base_output_dir, "images"), exist_ok=True)

        results_filepath = os.path.join(base_output_dir, f"{args.method_name}_evaluation_results.jsonl")
        prompt_map_path = os.path.join(base_output_dir, "prompt_map.json")
        
        dataset_path = f"dataset/{cur_dataset}"
        if is_main_process(rank):
            print(f"Loading dataset from: {dataset_path}")
        
        if cur_dataset == "geneval":
            dataset = GenevalPromptDataset(dataset_path, split="test")
            all_reward_scorers = {"geneval": 1.0}
            eval_batch_size = 14
        elif cur_dataset == "ocr":
            dataset = TextPromptDataset(dataset_path, split="test")
            all_reward_scorers = {"ocr": 1.0}
            eval_batch_size = 16
        elif cur_dataset == "pickscore":
            dataset = TextPromptDataset(dataset_path, split="test")
            all_reward_scorers = {
                "imagereward": 1.0,
                "pickscore": 1.0,
                "aesthetic": 1.0,
                "clipscore": 1.0,
                "hpsv2": 1.0,
            }
            eval_batch_size = 16
        elif cur_dataset == "drawbench":
            dataset = TextPromptDataset(dataset_path, split="test")
            all_reward_scorers = {
                "imagereward": 1.0,
                "pickscore": 1.0,
                "aesthetic": 1.0,
                "clipscore": 1.0,
                "hpsv2": 1.0,
            }
            eval_batch_size = 5
        else:
            if is_main_process(rank):
                print(f"Skipping unknown dataset: {cur_dataset}")
            continue

        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
        dataloader = DataLoader(
            dataset,
            batch_size=eval_batch_size,
            sampler=sampler,
            collate_fn=collate_fn,
            shuffle=False,
        )

        # --- Instantiate Reward Models ---
        if is_main_process(rank):
            print(f"Initializing reward models for {cur_dataset}...")
        scoring_fn = multi_score(device, all_reward_scorers)

        # --- Deterministic latent noise: per-rank generator, re-seeded per dataset ---
        generator = torch.Generator(device=device).manual_seed(args.seed + rank)

        # --- Evaluation Loop ---
        results_this_rank = []

        for batch in tqdm(dataloader, desc=f"Evaluating {cur_dataset} (Rank {rank})", disable=not is_main_process(rank)):
            prompts, metadata, indices = batch
            current_batch_size = len(prompts)

            with torch.cuda.amp.autocast(enabled=enable_amp, dtype=mixed_precision_dtype):
                with torch.no_grad():
                    images = pipeline(
                        prompts,
                        num_inference_steps=args.num_inference_steps,
                        guidance_scale=args.guidance_scale,
                        output_type="pt",
                        height=args.resolution,
                        width=args.resolution,
                        generator=generator,
                    )[0]
            
            all_scores, _ = scoring_fn(images, prompts, metadata, only_strict=False)
            
            for i in range(current_batch_size):
                sample_idx = indices[i]
                result_item = {
                    "sample_id": sample_idx,
                    "prompt": prompts[i],
                    "metadata": metadata[i] if metadata else {},
                    "scores": {},
                }

                if args.save_images:
                    sample_dir = os.path.join(base_output_dir, "images", f"{sample_idx:05d}")
                    os.makedirs(sample_dir, exist_ok=True)
                    file_name = f"{args.method_name}_{sample_idx:05d}.jpg"
                    image_path = os.path.join(sample_dir, file_name)

                    pil_image = Image.fromarray((images[i].cpu().float().numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
                    pil_image.save(image_path)
                    result_item["image_path"] = image_path
                
                for score_name, score_values in all_scores.items():
                    if isinstance(score_values, torch.Tensor):
                        result_item["scores"][score_name] = score_values[i].detach().cpu().item()
                    else:
                        result_item["scores"][score_name] = float(score_values[i])
                
                results_this_rank.append(result_item)
            
            del images, all_scores
        
        # --- Gather and Save Results ---
        dist.barrier()

        all_gathered_results = [None] * world_size
        dist.all_gather_object(all_gathered_results, results_this_rank)

        if is_main_process(rank):
            flat_results = [item for sublist in all_gathered_results for item in sublist]
            flat_results.sort(key=lambda x: x["sample_id"])

            with open(results_filepath, "w") as f_out:
                for result_item in flat_results:
                    f_out.write(json.dumps(result_item) + "\n")
            print(f"\nEvaluation {cur_dataset} finished. All {len(flat_results)} Results saved to {results_filepath}")

            if not os.path.exists(prompt_map_path):
                prompt_map = {
                    f"{item['sample_id']:05d}": item['prompt']
                    for item in flat_results
                }
                if args.save_images:
                    with open(prompt_map_path, "w", encoding="utf-8") as f_map:
                        json.dump(prompt_map, f_map, indent=4, ensure_ascii=False)
                    print(f"Prompt map saved to: {prompt_map_path}")
                
            all_scores_agg = defaultdict(list)
            for result in flat_results:
                for score_name, score_value in result["scores"].items():
                    if isinstance(score_value, (int, float)):
                        all_scores_agg[score_name].append(score_value)
            
            average_scores = {
                name: np.mean(list(filter(lambda score: score != -10.0, scores))) 
                for name, scores in all_scores_agg.items()
            }

            print(f"\n--- Average Scores ({cur_dataset}) ---")
            if not average_scores:
                print("No scores were found to average.")
            else:
                for name, avg_score in sorted(average_scores.items()):
                    print(f"{name:<20}: {avg_score:.4f}")
            print("----------------------")

            avg_scores_filepath = os.path.join(base_output_dir, f"{args.method_name}_average_scores.json")
            with open(avg_scores_filepath, "w") as f_avg:
                json.dump(average_scores, f_avg, indent=4)

        del scoring_fn
        del dataloader
        del dataset
        del results_this_rank
        torch.cuda.empty_cache()
        dist.barrier() 
    
    if is_main_process(rank):
        print("\nAll datasets evaluated successfully.")
        
    cleanup_distributed()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained diffusion model in a distributed manner.")
    parser.add_argument(
        "--pretrained_model",
        type=str,
        default=os.environ.get("SD3_MODEL", "stabilityai/stable-diffusion-3.5-medium"),
        help="Base model identifier or local Diffusers directory.",
    )
    parser.add_argument(
        "--lora_hf_path",
        type=str,
        default="",
        help="Huggingface path for LoRA.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="",
        help="Local path to the LoRA checkpoint directory (e.g., './save/run_name/checkpoints/checkpoint-5000').",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=["sd3"],
        help="Type of the base model ('sd3').",
    )
    parser.add_argument(
        "--dataset", type=str, nargs='+', required=True, choices=["geneval", "ocr", "pickscore", "drawbench"], help="Dataset type."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./evaluation_output_after",
        help="Directory to save evaluation results and generated images.",
    )
    parser.add_argument(
        "--num_inference_steps", type=int, default=40, help="Number of inference steps for the diffusion pipeline."
    )
    parser.add_argument("--guidance_scale", type=float, default=1.0, help="Classifier-free guidance scale.")
    parser.add_argument("--resolution", type=int, default=512, help="Resolution of the generated images.")
    parser.add_argument(
        "--save_images", action="store_true", help="Include this flag to save generated images to the output directory."
    )
    parser.add_argument(
        "--method_name", 
        type=str, 
        default="sd3_baseline", 
        help="Identifier for the method (e.g., 'FlowGRPO_Geneval', 'nft_Geneval')."
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help="Whether to use mixed precision. Choose between 'no', 'fp16', or 'bf16'.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducible latent noise."
    )

    args = parser.parse_args()
    main(args)
