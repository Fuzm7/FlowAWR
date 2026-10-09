from typing import Optional

import torch


def wan_pipeline_orig_awr(
    pipeline,
    prompt_embeds: torch.Tensor,
    negative_prompt_embeds: torch.Tensor,
    num_inference_steps: int,
    guidance_scale: float,
    height: int,
    width: int,
    num_frames: int,
    batch_size: int,
    device: torch.device,
    return_latents: bool = True,
    generator: Optional[torch.Generator] = None,
):
    """
    Wan2.1 official pipeline wrapper for AWR training.

    Uses the standard WanPipeline (UniPC solver) to ensure train-inference consistency.

    Args:
        return_latents: If True (sampling mode), returns (videos, latents_clean, initial_noise).
                        If False (eval mode), returns (videos,).
    """
    if return_latents:
        transformer_module = getattr(pipeline.transformer, 'module', pipeline.transformer)
        initial_noise = pipeline.prepare_latents(
            batch_size,
            transformer_module.config.in_channels,
            height,
            width,
            num_frames,
            torch.float32,
            device,
            generator=generator,
        )
        latents_clean = pipeline(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            output_type="latent",
            return_dict=False,
            num_frames=num_frames,
            height=height,
            width=width,
            latents=initial_noise.clone(),
        )[0]

        decode_input = latents_clean.to(pipeline.vae.dtype)
        latents_mean = (
            torch.tensor(pipeline.vae.config.latents_mean)
            .view(1, pipeline.vae.config.z_dim, 1, 1, 1)
            .to(decode_input.device, decode_input.dtype)
        )
        latents_std = (
            1.0 / torch.tensor(pipeline.vae.config.latents_std)
            .view(1, pipeline.vae.config.z_dim, 1, 1, 1)
            .to(decode_input.device, decode_input.dtype)
        )
        decode_input = decode_input / latents_std + latents_mean
        videos = pipeline.vae.decode(decode_input, return_dict=False)[0]
        videos = pipeline.video_processor.postprocess_video(videos, output_type="pt")

        return videos, latents_clean, initial_noise

    videos = pipeline(
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        output_type="pt",
        return_dict=False,
        num_frames=num_frames,
        height=height,
        width=width,
        generator=generator,
    )[0]

    return (videos,)
