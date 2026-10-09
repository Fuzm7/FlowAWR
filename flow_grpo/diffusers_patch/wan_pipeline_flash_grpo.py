"""Flash-GRPO 基准采样的 index-inject pipeline。

Flash-GRPO 的采样语义与本仓库 ``wan_pipeline_with_logprob``（前段 SDE + 后段
ODE）不同：它先走 ODE 前缀到选出注入点 ``index``，在 ``index`` 处注入一次
SDE 噪声并记录 log_prob，再走 ODE 后缀。训练侧用注入前 ``latents[index-1]``
与注入后 ``latents[index]`` 的 ``sde_step_with_logprob`` 计算新 log_prob，
与采样时存下的 ``log_probs[index]`` 求 PPO ratio。

``index`` 支持逐样本取值（形状 ``(B,)`` 的整型张量），也接受标量。Flash-GRPO
参照实现按 prompt 分配注入点后取 ``[0]`` 作用于全 batch，其前提是每卡恒为同一
prompt 的若干副本（``train_batch_size=1`` + ``num_videos_per_prompt``）。本仓库
的 OOD 采样器会把不同 prompt 混入同一卡，故必须按样本各自注入：否则训练侧
``timesteps.gather(1, chosen_idx)`` 取到的时间步与采样时的实际注入步不符，
PPO ratio 失去意义。逐样本实现按 ``num_inference_steps`` 步循环，每步对全 batch
做一次前传，注入子集走 SDE、其余走 ODE，前传次数与共享 index 的实现相同。

``compute_coe_table`` 计算 Flash-GRPO 的 per-timestep 归一化系数
``coe = 1 / (sqrt(-dt)/std_dev_t + std_dev_t*sqrt(-dt)*(1-sigma)/(2*sigma))``，
即 Flash-GRPO 训练 loss 中 ``value_dict`` 的运行时来源。探针验证该值对
20-step Wan2.1 grid 的 999/982/963/944/922/899/874/847/817/785 与参照硬编码
dict bit 级一致。
"""

from typing import Optional, Union

import torch


def compute_coe_table(scheduler, num_inference_steps: int):
    """按 scheduler 的 sigmas 计算前 ``num_inference_steps`` 步的 coe 表。

    返回 ``{int(timestep): float(coe)}``，仅覆盖 Flash-GRPO 权重采样覆盖的
    时间步范围（首 10 步）。coe 定义见模块 docstring。
    """
    scheduler.set_timesteps(num_inference_steps)
    sigmas = scheduler.sigmas
    timesteps = scheduler.timesteps
    sigma_min = sigmas[-1].item()
    sigma_max = sigmas[1].item()
    table = {}
    for i in range(len(timesteps)):
        sigma = sigmas[i].item()
        sigma_prev = sigmas[i + 1].item()
        dt = sigma_prev - sigma
        std_dev_t = sigma_min + (sigma_max - sigma_min) * sigma
        sqrt_dt = torch.sqrt(torch.as_tensor(-1.0 * dt)).item()
        coe = 1.0 / (
            sqrt_dt / std_dev_t
            + (std_dev_t * sqrt_dt * (1.0 - sigma)) / (2.0 * sigma)
        )
        table[int(timesteps[i].item())] = coe
    return table


def _inject_step(scheduler, model_output, timestep, sample, generator, determistic):
    """单步 index 注入：复用本仓库 ``sde_step_with_logprob``。

    返回注入后 latent、log_prob、std_dev_t、sqrt_dt。
    """
    from flow_grpo.diffusers_patch.wan_pipeline_with_logprob import (
        sde_step_with_logprob,
    )

    prev_sample, log_prob, _, std_dev_t, sqrt_dt = sde_step_with_logprob(
        scheduler,
        model_output,
        timestep,
        sample,
        generator=generator,
        determistic=determistic,
        return_dt_and_std_dev_t=True,
    )
    return prev_sample, log_prob, std_dev_t, sqrt_dt


def _normalize_index(index, batch_size, num_inference_steps, device):
    """把标量或张量形式的 ``index`` 归一为形状 ``(B,)`` 的 long 张量。

    标量按全 batch 广播，保持与共享 index 调用方（eval 路径）的兼容。
    """
    if isinstance(index, torch.Tensor):
        index_tensor = index.to(device=device, dtype=torch.long).reshape(-1)
        if index_tensor.numel() == 1:
            index_tensor = index_tensor.expand(batch_size).clone()
        assert index_tensor.numel() == batch_size, (
            f"index 长度 {index_tensor.numel()} 与 batch_size {batch_size} 不符"
        )
    else:
        index_tensor = torch.full(
            (batch_size,), int(index), device=device, dtype=torch.long
        )
    out_of_range = (index_tensor < 0) | (index_tensor >= num_inference_steps)
    if bool(out_of_range.any()):
        raise ValueError(
            f"index 必须落在 [0, num_inference_steps-1]，"
            f"越界取值={index_tensor[out_of_range].tolist()} "
            f"num_inference_steps={num_inference_steps}"
        )
    return index_tensor


def wan_pipeline_flash_grpo(
    self,
    prompt_embeds: torch.Tensor,
    negative_prompt_embeds: torch.Tensor,
    num_inference_steps: int,
    guidance_scale: float,
    height: int,
    width: int,
    num_frames: int,
    batch_size: int,
    device: torch.device,
    index,
    generator: Optional[torch.Generator] = None,
    attention_kwargs: Optional[dict] = None,
):
    """Flash-GRPO 的 index-inject 采样。

    返回 ``(videos, latents_pair, log_probs, index_tensor)``，其中
    ``latents_pair`` 为长度 2 的列表 ``[latents_before, latents_after]``
    （各样本在其自身注入步的前后 latent），``log_probs`` 为长度 1 的列表
    （各样本注入步的 log_prob）。``index`` 可为标量或形状 ``(B,)`` 的整型
    张量，取值须落在 ``[0, num_inference_steps-1]``（0 对应 timestep=999
    的最高噪步，与 Flash-GRPO 参照的 ``weights=[1]*10`` 下界一致）。
    """
    self.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = self.scheduler.timesteps

    self._guidance_scale = guidance_scale
    self._attention_kwargs = attention_kwargs
    self._current_timestep = None

    index_tensor = _normalize_index(
        index, batch_size, num_inference_steps, device
    )

    transformer_module = getattr(self.transformer, "module", self.transformer)
    latents = self.prepare_latents(
        batch_size,
        transformer_module.config.in_channels,
        height,
        width,
        num_frames,
        torch.float32,
        device,
        generator=generator,
    )

    def _predict_noise(current_latents, timestep, cond_embeds, uncond_embeds):
        model_timestep = timestep.expand(current_latents.shape[0])
        noise_pred = self.transformer(
            hidden_states=current_latents,
            timestep=model_timestep,
            encoder_hidden_states=cond_embeds,
            attention_kwargs=self._attention_kwargs,
            return_dict=False,
        )[0]
        if self.do_classifier_free_guidance and uncond_embeds is not None:
            noise_uncond = self.transformer(
                hidden_states=current_latents,
                timestep=model_timestep,
                encoder_hidden_states=uncond_embeds,
                attention_kwargs=self._attention_kwargs,
                return_dict=False,
            )[0]
            noise_pred = noise_uncond + guidance_scale * (noise_pred - noise_uncond)
        return noise_pred

    # 每个样本恰在自身 index 步注入一次：其余步走确定性 ODE。三个 (B, ...)
    # 缓冲按注入步逐样本填充，故不同样本可取不同注入点。
    latents_before = torch.zeros_like(latents)
    latents_after = torch.zeros_like(latents)
    injected_log_prob = torch.zeros(batch_size, device=device, dtype=torch.float32)

    with self.progress_bar(total=num_inference_steps) as progress_bar:
        for i in range(num_inference_steps):
            t = timesteps[i]
            self._current_timestep = t
            noise_pred = _predict_noise(latents, t, prompt_embeds, negative_prompt_embeds)
            ode_latents, _, _, _ = _inject_step(
                self.scheduler, noise_pred.float(), t.unsqueeze(0),
                latents.float(), None, True,
            )

            inject_mask = index_tensor == i
            if not bool(inject_mask.any()):
                latents = ode_latents
                progress_bar.update()
                continue

            sde_latents, step_log_prob, _, _ = _inject_step(
                self.scheduler, noise_pred.float(), t.unsqueeze(0),
                latents.float(), generator, False,
            )
            mask_view = inject_mask.view(-1, *([1] * (latents.ndim - 1)))
            latents_before = torch.where(mask_view, latents.float(), latents_before)
            latents_after = torch.where(mask_view, sde_latents, latents_after)
            injected_log_prob = torch.where(
                inject_mask, step_log_prob, injected_log_prob
            )
            latents = torch.where(mask_view, sde_latents, ode_latents)
            progress_bar.update()

    self._current_timestep = None

    latents_decode = latents.to(self.vae.dtype)
    latents_mean = (
        torch.tensor(self.vae.config.latents_mean)
        .view(1, self.vae.config.z_dim, 1, 1, 1)
        .to(latents_decode.device, latents_decode.dtype)
    )
    latents_std = (
        1.0 / torch.tensor(self.vae.config.latents_std)
        .view(1, self.vae.config.z_dim, 1, 1, 1)
        .to(latents_decode.device, latents_decode.dtype)
    )
    latents_decode = latents_decode / latents_std + latents_mean
    videos = self.vae.decode(latents_decode, return_dict=False)[0]
    videos = self.video_processor.postprocess_video(videos, output_type="pt")

    return videos, [latents_before, latents_after], [injected_log_prob], index_tensor
