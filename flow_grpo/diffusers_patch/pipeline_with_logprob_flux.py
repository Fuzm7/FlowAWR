from typing import Any

from .flux_pipeline_orig_awr import flux_pipeline_orig_awr
from .pipeline_with_logprob import pipeline_with_logprob


FLUX_SOLVERS = ("orig", "dpm2")


def validate_flux_solver(solver: str, deterministic: bool) -> str:
    if solver not in FLUX_SOLVERS:
        raise ValueError(
            f"config.sample.solver must be one of {FLUX_SOLVERS}, got {solver!r}"
        )
    if solver == "dpm2" and not deterministic:
        raise ValueError(
            "config.sample.deterministic must be True when config.sample.solver='dpm2'"
        )
    return solver


def pipeline_with_logprob_flux(
    pipeline: Any,
    *,
    solver: str,
    return_latents: bool,
    prompt_embeds: Any,
    pooled_prompt_embeds: Any,
    num_inference_steps: int,
    guidance_scale: float,
    height: int,
    width: int,
    batch_size: int,
    device: Any,
    noise_level: float,
    deterministic: bool,
) -> tuple:
    solver = validate_flux_solver(solver, deterministic)

    if solver == "orig":
        return flux_pipeline_orig_awr(
            pipeline,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            batch_size=batch_size,
            device=device,
            return_latents=return_latents,
        )

    images, all_latents, latent_image_ids, _, _ = pipeline_with_logprob(
        pipeline,
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        output_type="pt",
        height=height,
        width=width,
        noise_level=noise_level,
        deterministic=deterministic,
        solver="dpm2",
        model_type="flux",
    )
    if return_latents:
        return images, all_latents[-1], latent_image_ids
    return (images,)
