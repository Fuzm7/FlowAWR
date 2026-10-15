import os
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "videoalign"))
from inference import VideoVLMRewardInference


def load_videoalign():
    for name in ("VIDEOALIGN_CHECKPOINT_DIR", "VIDEOALIGN_BASE_MODEL_PATH"):
        if not os.environ.get(name):
            raise ValueError(f"Set {name} to a local model directory.")
    checkpoint_dir = Path(os.environ["VIDEOALIGN_CHECKPOINT_DIR"]).expanduser().resolve()
    base_model_path = Path(os.environ["VIDEOALIGN_BASE_MODEL_PATH"]).expanduser().resolve()
    if not base_model_path.is_dir():
        raise FileNotFoundError(f"Base model directory not found: {base_model_path}")
    if not (checkpoint_dir / "model_config.json").is_file():
        raise FileNotFoundError(f"model_config.json not found in {checkpoint_dir}")
    checkpoints = [p for p in checkpoint_dir.glob("checkpoint-*") if p.is_dir()]
    if not checkpoints or any(not p.name.removeprefix("checkpoint-").isdigit() for p in checkpoints):
        raise ValueError("VideoAlign requires checkpoint directories named checkpoint-<integer>.")
    latest = max(checkpoints, key=lambda p: int(p.name.removeprefix("checkpoint-")))
    if not (latest / "model.pth").is_file() and not all(
        (latest / name).is_file() for name in ("adapter_model.safetensors", "non_lora_state_dict.pth")
    ):
        raise FileNotFoundError(f"Incomplete VideoAlign checkpoint: {latest}")

    inferencer = VideoVLMRewardInference(
        str(checkpoint_dir), device="cuda", dtype=torch.bfloat16,
        model_name_or_path=str(base_model_path),
    )
    torch.cuda.empty_cache()

    @torch.no_grad()
    def compute_videoalign(video, prompts):
        rewards = inferencer.reward(video, prompts, use_norm=True)
        return rewards

    return compute_videoalign
