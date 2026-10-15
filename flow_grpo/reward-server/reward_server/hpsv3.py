import os
from pathlib import Path

import torch
from hpsv3 import HPSv3RewardInferencer


def load_hpsv3score():
    for name in ("HPSV3_CHECKPOINT_PATH", "HPSV3_BASE_MODEL_PATH"):
        if not os.environ.get(name):
            raise ValueError(f"Set {name} to a local model path.")
    checkpoint_path = Path(os.environ["HPSV3_CHECKPOINT_PATH"]).expanduser().resolve()
    base_model_path = Path(os.environ["HPSV3_BASE_MODEL_PATH"]).expanduser().resolve()
    config_path = Path(os.environ.get(
        "HPSV3_CONFIG_PATH",
        Path(__file__).resolve().parents[2] / "HPSv3/hpsv3/config/HPSv3_7B.yaml",
    )).expanduser().resolve()
    for path in (checkpoint_path, config_path):
        if not path.is_file():
            raise FileNotFoundError(f"Model file not found: {path}")
    if not base_model_path.is_dir():
        raise FileNotFoundError(f"Base model directory not found: {base_model_path}")

    inferencer = HPSv3RewardInferencer(
        config_path=str(config_path), device="cuda", checkpoint_path=str(checkpoint_path),
        model_name_or_path=str(base_model_path),
    )
    torch.cuda.empty_cache()

    @torch.no_grad()
    def compute_hpsv3score(images, prompts):
        rewards = inferencer.reward(images, prompts)
        return [reward[0].item() for reward in rewards]

    return compute_hpsv3score
