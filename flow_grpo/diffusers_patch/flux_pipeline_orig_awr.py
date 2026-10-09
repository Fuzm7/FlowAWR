from typing import Optional

import torch


def flux_pipeline_orig_awr(
    pipeline,
    prompt_embeds: torch.Tensor,
    pooled_prompt_embeds: torch.Tensor,
    num_inference_steps: int,
    guidance_scale: float,
    height: int,
    width: int,
    batch_size: int,
    device: torch.device,
    return_latents: bool = True,
    generator: Optional[torch.Generator] = None,
):
    """
    FLUX official pipeline wrapper for AWR training.

    Uses the standard FluxPipeline (FlowMatchEulerDiscreteScheduler with dynamic
    shifting) to ensure train-inference consistency.

    Args:
        return_latents: If True (sampling mode), returns (images, latents_clean, latent_image_ids).
                        latents_clean is in FLUX packed layout (B, (H/16)*(W/16), 64).
                        If False (eval mode), returns (images,).
    """
    if return_latents:
        transformer_module = getattr(pipeline.transformer, "module", pipeline.transformer)
        num_channels_latents = transformer_module.config.in_channels // 4
        initial_noise, latent_image_ids = pipeline.prepare_latents(
            batch_size,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            None,
        )
        latents_clean = pipeline(
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            output_type="latent",
            return_dict=False,
            height=height,
            width=width,
            latents=initial_noise.clone(),
        )[0]

        # Decode path replicated from FluxPipeline.__call__ (output_type != "latent" branch).
        decode_input = pipeline._unpack_latents(latents_clean, height, width, pipeline.vae_scale_factor)
        decode_input = (decode_input / pipeline.vae.config.scaling_factor) + pipeline.vae.config.shift_factor
        decode_input = decode_input.to(pipeline.vae.dtype)
        images = pipeline.vae.decode(decode_input, return_dict=False)[0]
        images = pipeline.image_processor.postprocess(images, output_type="pt")

        return images, latents_clean, latent_image_ids

    images = pipeline(
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        output_type="pt",
        return_dict=False,
        height=height,
        width=width,
        generator=generator,
    )[0]

    return (images,)
