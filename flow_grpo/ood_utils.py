"""OOD 混合训练共享组件。

本模块抽取自定义 sampler 与 OOD 数据工具，供 AWR OOD 脚本与 Flash-GRPO OOD
脚本复用。所有组件仅依赖 torch / numpy / cv2 / 标准库，不依赖任何训练脚本，
以避免循环引用。
"""

import json
import math
import os
import random
from collections import defaultdict

import cv2
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, Sampler
from diffusers.models.transformers.transformer_wan import WanTransformerBlock

from flow_grpo.diffusers_patch.wan_prompt_embedding import encode_prompt


def plan_per_batch(total_units, num_batches):
    """把 ``total_units`` 个单位尽量均匀分到 ``num_batches`` 个批次。

    余数分配给前几个批次，各批次至多相差 1 个单位。当单位数少于批次数时，
    前 ``total_units`` 个批次各得 1 个单位，其余批次为 0。
    """
    base, remainder = divmod(total_units, num_batches)
    return [base + 1 if b < remainder else base for b in range(num_batches)]


class DistributedKRepeatEpochSampler(Sampler):
    """按 epoch 规划域内 prompt，允许各采样批次的 prompt 数不等。

    每个 prompt 仍占用同一批次内的 ``k`` 个连续 slot（k 个 repeat 不跨批次）。
    每批次的 prompt 数取 ``step = num_replicas / gcd(num_replicas, k)`` 的倍数，
    使该批次的 ``m_b * k`` 可被卡数整除。相比固定每批次 prompt 数的
    ``DistributedKRepeatSampler``，本类把整除约束从「每批次」放宽到「每 epoch」，
    从而允许 ``m_epoch`` 取更细的值。

    ``set_epoch`` 接收全局批次计数 ``epoch * num_batches + batch_index``。
    """

    def __init__(self, dataset, m_epoch, k, num_replicas, rank, num_batches, seed=0):
        step = num_replicas // math.gcd(num_replicas, k)
        assert m_epoch % step == 0, (
            f"域内 prompt 数 m_epoch={m_epoch} 须为 {step} 的倍数 "
            f"(= world_size{num_replicas}/gcd(world_size,k{k}))"
        )
        assert m_epoch // step >= num_batches, (
            f"域内 prompt 数不足以分配到每个批次: m_epoch={m_epoch} "
            f"仅含 {m_epoch // step} 个 {step}-单位 < num_batches={num_batches}"
        )
        assert len(dataset) >= m_epoch, (
            f"域内 prompt 池不足: len(dataset)={len(dataset)} < m_epoch={m_epoch}"
        )
        self.dataset = dataset
        self.m_epoch = m_epoch
        self.k = k
        self.num_replicas = num_replicas
        self.rank = rank
        self.num_batches = num_batches
        self.seed = seed
        self.step = step
        self.prompt_counts = [
            units * step for units in plan_per_batch(m_epoch // step, num_batches)
        ]
        self.sample_counts = [
            count * k // num_replicas for count in self.prompt_counts
        ]
        self.epoch = 0

    def __iter__(self):
        while True:
            epoch_id, batch_id = divmod(self.epoch, self.num_batches)
            g = torch.Generator()
            g.manual_seed(self.seed + epoch_id)
            epoch_prompts = torch.randperm(
                len(self.dataset), generator=g
            )[:self.m_epoch].tolist()

            start_prompt = sum(self.prompt_counts[:batch_id])
            batch_prompts = epoch_prompts[
                start_prompt:start_prompt + self.prompt_counts[batch_id]
            ]
            repeated = [idx for idx in batch_prompts for _ in range(self.k)]
            perm = torch.randperm(len(repeated), generator=g).tolist()
            repeated = [repeated[i] for i in perm]
            count = self.sample_counts[batch_id]
            start = self.rank * count
            yield repeated[start:start + count]

    def set_epoch(self, epoch):
        self.epoch = epoch


class DistributedOODSampler(Sampler):
    """为域外 prompt 同时分配 rollout slot 与 offline slot。

    ``m_ood`` 为每 epoch 的域外 prompt 数。两种模式共用 epoch 级规划：prompt 与
    offline slot 按 epoch 规划后分摊到各批次，各批次数量至多相差一个分配单位；
    由于奖励统计按 prompt 对拼接后的全体样本归组，offline 无需与其 prompt 的
    rollout 落在同一批次。

    ``group_mode="replace"``：每个被选中的 prompt 占用 ``k`` 个组内 slot，其中
    ``k - n_ood`` 个走 on-policy rollout，``n_ood`` 个取自该 prompt 的域外视频。

    ``group_mode="append"``：每个被选中的 prompt 拿满 ``k`` 个 rollout slot，
    ``n_ood`` 个域外视频作为额外样本追加。

    每次迭代产出本 rank 的 ``(rollout_prompt_indices, offline_slots)``，其中
    ``offline_slots`` 的元素为 ``(prompt_index, replica_id)``，``replica_id``
    直接作为 ``ood_entries`` 的下标。
    """

    def __init__(
        self, dataset, m_ood, k, n_ood, num_replicas, rank, seed=0,
        group_mode="replace", num_batches=1,
    ):
        assert group_mode in ("replace", "append"), (
            f"ood_group_mode 必须为 'replace' 或 'append'，得到 {group_mode!r}"
        )
        if group_mode == "replace":
            assert 1 <= n_ood < k, (
                f"replace 模式下 n_ood 必须落在 [1, k-1]，得到 n_ood={n_ood} k={k}"
            )
        else:
            assert n_ood >= 1, f"append 模式下 n_ood 必须 >= 1，得到 n_ood={n_ood}"
        self.dataset = dataset
        self.m_ood = m_ood
        self.k = k
        self.n_ood = n_ood
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.group_mode = group_mode
        self.num_batches = num_batches
        self.rollout_per_prompt = k if group_mode == "append" else k - n_ood

        step = num_replicas // math.gcd(num_replicas, self.rollout_per_prompt)
        assert m_ood % step == 0, (
            f"{group_mode} 模式下每 epoch 域外 prompt 数 m_ood={m_ood} 须为 "
            f"{step} 的倍数 (= world_size{num_replicas}/"
            f"gcd(world_size,rollout_per_prompt{self.rollout_per_prompt}))"
        )
        assert m_ood // step >= 1, (
            f"域外 prompt 数不足一个分配单位: m_ood={m_ood} < step={step}"
        )
        assert len(dataset) >= m_ood, (
            f"OOD prompt 池不足: len(dataset)={len(dataset)} < m_ood={m_ood}"
        )
        self.offline_total = m_ood * n_ood
        assert self.offline_total % num_replicas == 0, (
            f"每 epoch 域外样本数不可被卡数整除: m_ood{m_ood}*n_ood{n_ood}"
            f"={self.offline_total}, num_replicas={num_replicas}"
        )
        self.step = step
        self.prompt_counts = [
            units * step
            for units in plan_per_batch(m_ood // step, num_batches)
        ]
        self.rollout_counts = [
            count * self.rollout_per_prompt // num_replicas
            for count in self.prompt_counts
        ]
        self.offline_counts = plan_per_batch(
            self.offline_total // num_replicas, num_batches
        )
        self.rollout_total = m_ood * self.rollout_per_prompt
        self.rollout_per_rank = max(self.rollout_counts)
        self.offline_per_rank = max(self.offline_counts)
        self.epoch = 0

    def _epoch_iter(self):
        epoch_id, batch_id = divmod(self.epoch, self.num_batches)
        g = torch.Generator()
        g.manual_seed(self.seed + epoch_id)
        epoch_prompts = torch.randperm(
            len(self.dataset), generator=g
        )[:self.m_ood].tolist()

        start_prompt = sum(self.prompt_counts[:batch_id])
        batch_prompts = epoch_prompts[
            start_prompt:start_prompt + self.prompt_counts[batch_id]
        ]
        rollout_slots = [
            idx for idx in batch_prompts for _ in range(self.rollout_per_prompt)
        ]
        rollout_perm = torch.randperm(len(rollout_slots), generator=g).tolist()
        rollout_slots = [rollout_slots[i] for i in rollout_perm]
        rollout_count = self.rollout_counts[batch_id]
        rollout_start = self.rank * rollout_count

        # offline slot 的置换必须与 batch_id 无关，否则各批次切到的区间虽不重叠，
        # 映射回的原始 slot 会重复。故用独立 generator，不复用被 rollout 消耗过的 g。
        g_off = torch.Generator()
        g_off.manual_seed(self.seed + epoch_id + 10 ** 6)
        all_offline = [
            (idx, replica_id)
            for idx in epoch_prompts
            for replica_id in range(self.n_ood)
        ]
        offline_perm = torch.randperm(len(all_offline), generator=g_off).tolist()
        all_offline = [all_offline[i] for i in offline_perm]
        count = self.offline_counts[batch_id]
        batch_start = sum(self.offline_counts[:batch_id]) * self.num_replicas
        offline_start = batch_start + self.rank * count
        return (
            rollout_slots[rollout_start:rollout_start + rollout_count],
            all_offline[offline_start:offline_start + count],
        )

    def __iter__(self):
        while True:
            yield self._epoch_iter()

    def set_epoch(self, epoch):
        self.epoch = epoch


class UnifiedOODSampler(Sampler):
    """replace 模式统一采样器。

    ``m_epoch`` 个 prompt（域内 + 域外混合）各占 ``k`` 个 slot：域内 prompt 的
    ``k`` 个 slot 全为 rollout；域外 prompt 的 ``k`` 个 slot 中 ``k - n_ood`` 个为
    rollout、``n_ood`` 个为 offline（读 mp4 + VAE encode）。每个 rank 每批次恒定
    拿到 ``batch_size`` 个 slot（rollout 在前、offline 在后）。slot 池按 epoch
    构建、洗牌、切块，因此 ``m_epoch * k`` 必须能被 ``num_replicas * batch_size``
    整除——这是 replace 模式下唯一的分割约束，与 ood_ratio / n_ood 无关。

    yield 契约为 ``(rollout_slots, offline_slots)``，每个 slot 为
    ``(kind, prompt_index, replica_id)``，其中 ``kind`` 取 ``KIND_PURE`` /
    ``KIND_OOD_ROLLOUT`` / ``KIND_OOD_OFFLINE``。offline slot 的 ``replica_id``
    为该 prompt 的 ood_entries 下标，其余 slot 的 ``replica_id`` 恒为 0。
    """

    KIND_PURE = 0
    KIND_OOD_ROLLOUT = 1
    KIND_OOD_OFFLINE = 2

    def __init__(
        self, pure_dataset, ood_dataset, m_epoch, k, n_ood, m_pure,
        num_replicas, rank, num_batches, seed=0,
    ):
        assert m_epoch * k % (num_replicas * num_batches) == 0, (
            f"replace 模式 slot 池不可分割: m_epoch{m_epoch}*k{k}="
            f"{m_epoch * k}, num_replicas{num_replicas}*num_batches"
            f"{num_batches}={num_replicas * num_batches}"
        )
        self.pure_dataset = pure_dataset
        self.ood_dataset = ood_dataset
        self.m_epoch = m_epoch
        self.k = k
        self.n_ood = n_ood
        self.m_pure = m_pure
        self.m_ood = m_epoch - m_pure
        self.num_replicas = num_replicas
        self.rank = rank
        self.num_batches = num_batches
        self.seed = seed
        self.batch_size = m_epoch * k // (num_replicas * num_batches)
        # 每卡每 epoch 的 clean latent 总量：全部 slot 均分到各卡。
        self.per_rank_clean_total = m_epoch * k // num_replicas
        self.rollout_total = m_pure * k + self.m_ood * (k - n_ood)
        self.offline_total = self.m_ood * n_ood
        self.epoch = 0
        # zone 划分把 prompt 池对半切，两个 zone 分别喂给一个 optimizer step，
        # 故 pure/ood prompt 数与块数均须为偶数。
        assert m_pure % 2 == 0, f"zone 划分要求域内 prompt 数为偶数，得到 m_pure={m_pure}"
        assert self.m_ood % 2 == 0, (
            f"zone 划分要求域外 prompt 数为偶数，得到 m_ood={self.m_ood}"
        )
        num_blocks = num_replicas * num_batches
        assert num_blocks % 2 == 0, (
            f"zone 划分要求块数为偶数: num_replicas{num_replicas}*num_batches"
            f"{num_batches}={num_blocks}"
        )
        half_blocks = num_blocks // 2
        zone_rollout = m_pure // 2 * k + self.m_ood // 2 * (k - n_ood)
        assert zone_rollout >= half_blocks, (
            f"单 zone 的 rollout slot 数 {zone_rollout} < 半区块数 {half_blocks}，"
            "无法保证每块至少 1 个 rollout"
        )

    def _build_slots(self):
        g_pure = torch.Generator()
        g_pure.manual_seed(self.seed + self.epoch)
        pure_prompts = torch.randperm(
            len(self.pure_dataset), generator=g_pure
        )[:self.m_pure].tolist()

        g_ood = torch.Generator()
        g_ood.manual_seed(self.seed + (1 << 20) + self.epoch)
        ood_prompts = torch.randperm(
            len(self.ood_dataset), generator=g_ood
        )[:self.m_ood].tolist()

        # zone 划分：pure 与 ood prompt 各对半切，zone z 的 slot 只铺到块号
        # [z*half_blocks, (z+1)*half_blocks)。训练侧按同一边界切分批次，故每个
        # optimizer step 恰好覆盖一个 zone，即 m_ood/2 个完整域外 prompt。
        num_blocks = self.num_replicas * self.num_batches
        half_blocks = num_blocks // 2
        pure_zones = (
            pure_prompts[:self.m_pure // 2], pure_prompts[self.m_pure // 2:],
        )
        ood_zones = (
            ood_prompts[:self.m_ood // 2], ood_prompts[self.m_ood // 2:],
        )

        slots = []
        for zone, (zone_pure, zone_ood) in enumerate(zip(pure_zones, ood_zones)):
            rollout_slots = [
                (self.KIND_PURE, idx, 0)
                for idx in zone_pure for _ in range(self.k)
            ]
            rollout_slots += [
                (self.KIND_OOD_ROLLOUT, idx, 0)
                for idx in zone_ood for _ in range(self.k - self.n_ood)
            ]
            offline_slots = [
                (self.KIND_OOD_OFFLINE, idx, rid)
                for idx in zone_ood for rid in range(self.n_ood)
            ]

            # 每个 zone 用独立 generator 偏移，两 zone 不共享置换序列。
            g_ro = torch.Generator()
            g_ro.manual_seed(self.seed + (1 << 21) + self.epoch + zone * (1 << 10))
            rollout_perm = torch.randperm(len(rollout_slots), generator=g_ro).tolist()
            rollout_slots = [rollout_slots[i] for i in rollout_perm]

            g_off = torch.Generator()
            g_off.manual_seed(self.seed + (1 << 22) + self.epoch + zone * (1 << 10))
            offline_perm = torch.randperm(len(offline_slots), generator=g_off).tolist()
            offline_slots = [offline_slots[i] for i in offline_perm]

            # 两级布局：先给本 zone 的每块铺 1 个 rollout，再把剩余 rollout 与
            # offline 依次轮转填入。cursor 单调递增且 rollout 段先填，故每块内
            # rollout 天然排在 offline 之前。
            blocks = [[] for _ in range(half_blocks)]
            for bi in range(half_blocks):
                blocks[bi].append(rollout_slots[bi])
            cursor = half_blocks
            for slot in rollout_slots[half_blocks:]:
                blocks[cursor % half_blocks].append(slot)
                cursor += 1
            for slot in offline_slots:
                blocks[cursor % half_blocks].append(slot)
                cursor += 1

            for block in blocks:
                assert block[0][0] != self.KIND_OOD_OFFLINE, (
                    "replace 模式每块须至少 1 个 rollout slot 且排在首位"
                )
            slots += [s for block in blocks for s in block]

        assert len(slots) == self.m_epoch * self.k, (
            f"slot 池大小 {len(slots)} != m_epoch*k = {self.m_epoch * self.k}"
        )
        return slots

    def _block_range(self, batch_index):
        start = (
            batch_index * self.num_replicas + self.rank
        ) * self.batch_size
        return start, start + self.batch_size

    def __iter__(self):
        while True:
            slots = self._build_slots()
            for b in range(self.num_batches):
                start, end = self._block_range(b)
                rollout = [
                    slots[p] for p in range(start, end)
                    if slots[p][0] != self.KIND_OOD_OFFLINE
                ]
                offline = [
                    slots[p] for p in range(start, end)
                    if slots[p][0] == self.KIND_OOD_OFFLINE
                ]
                assert len(rollout) >= 1, "replace 模式每批次须至少 1 个 rollout slot"
                assert len(rollout) + len(offline) == self.batch_size, (
                    f"块大小 {len(rollout) + len(offline)} != {self.batch_size}"
                )
                yield rollout, offline

    def set_epoch(self, epoch):
        """设置全局批次计数 ``epoch * num_batches + batch_index``。

        slot 池在 ``__iter__`` 中每 ``num_batches`` 次 yield 后重建一次，故一个
        epoch 内只有首次调用（``batch_index == 0``）的取值参与池构建，其余调用
        不改变本 epoch 的采样结果。
        """
        self.epoch = epoch


def load_ood_latents(pipeline, video_paths, height, width, num_frames, device):
    """把域外 mp4 编码为与 rollout ``latents_clean`` 同分布的 latent。

    规范化沿用 ``wan_pipeline_orig_awr`` 的约定，即 ``(raw - mean) * (1 / std)``，
    与其 decode 前的 ``latents / (1 / std) + mean`` 互为逆变换。
    ``video_paths`` 为空时返回形状 ``(0, z_dim, ...)`` 的空张量，以支持某些采样
    批次不含域外样本的配额分配。
    """
    if not video_paths:
        latent_frames = (num_frames - 1) // pipeline.vae_scale_factor_temporal + 1
        return torch.empty(
            0,
            pipeline.vae.config.z_dim,
            latent_frames,
            height // pipeline.vae_scale_factor_spatial,
            width // pipeline.vae_scale_factor_spatial,
            device=device,
            dtype=torch.float32,
        )
    latents_mean = (
        torch.tensor(pipeline.vae.config.latents_mean)
        .view(1, pipeline.vae.config.z_dim, 1, 1, 1)
        .to(device, torch.float32)
    )
    inv_latents_std = (
        1.0 / torch.tensor(pipeline.vae.config.latents_std)
        .view(1, pipeline.vae.config.z_dim, 1, 1, 1)
        .to(device, torch.float32)
    )

    latents = []
    for video_path in video_paths:
        assert os.path.exists(video_path), f"OOD 视频不存在: {video_path}"
        capture = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        capture.release()
        assert len(frames) == num_frames, (
            f"OOD 视频帧数必须为 {num_frames}，实际读到 {len(frames)}: {video_path}"
        )
        pixel_values = np.stack(frames).astype(np.float32) / 255.0
        pixel_values = pipeline.video_processor.preprocess_video(
            pixel_values, height=height, width=width
        ).to(device, dtype=pipeline.vae.dtype)
        with torch.no_grad():
            raw_latent = pipeline.vae.encode(pixel_values).latent_dist.mode().float()
        latents.append((raw_latent - latents_mean) * inv_latents_std)
        del pixel_values, raw_latent

    return torch.cat(latents, dim=0)


def compute_ood_group_stats(prompts, rewards, ood_mask):
    """统计域外样本在组内相对 on-policy 样本的胜率与领先幅度。

    仅统计同时含域外与 on-policy 样本的组。三个胜率的随机基线均为 0.5，
    完全分离时为 1.0。奖励取重打分后的标量，即进入 advantage 计算的同一数值。
    """
    prompt_array = np.array(prompts)
    pairwise_rates = []
    topn_rates = []
    group_wins = []
    margins = []
    for prompt in np.unique(prompt_array):
        in_group = prompt_array == prompt
        ood_scores = rewards[in_group & ood_mask]
        rollout_scores = rewards[in_group & ~ood_mask]
        if ood_scores.size == 0 or rollout_scores.size == 0:
            continue
        pairwise_rates.append(
            float((ood_scores[:, None] > rollout_scores[None, :]).mean())
        )
        group_scores = np.concatenate([ood_scores, rollout_scores])
        top_ranked = np.argsort(-group_scores)[:ood_scores.size]
        topn_rates.append(float((top_ranked < ood_scores.size).mean()))
        group_wins.append(float(ood_scores.mean() > rollout_scores.mean()))
        margins.append(float(ood_scores.mean() - rollout_scores.mean()))

    if not pairwise_rates:
        return {}
    return {
        "ood_win_rate_pairwise": float(np.mean(pairwise_rates)),
        "ood_win_rate_topn": float(np.mean(topn_rates)),
        "ood_win_rate_group": float(np.mean(group_wins)),
        "ood_margin_mean": float(np.mean(margins)),
        "ood_group_count": len(pairwise_rates),
    }


def compute_ood_exit_mask(prompts, rewards, ood_mask):
    """标记应退出训练的域外样本。

    组内 on-policy 样本的 reward 均值严格大于域外样本均值时，判定该组自身
    rollout 已优于域外，其域外样本剔除出本 epoch 训练。
    """
    prompt_array = np.array(prompts)
    exit_mask = np.zeros_like(ood_mask, dtype=bool)
    for prompt in np.unique(prompt_array):
        in_group = prompt_array == prompt
        group_ood = in_group & ood_mask
        rollout_scores = rewards[in_group & ~ood_mask]
        if not group_ood.any() or rollout_scores.size == 0:
            continue
        ood_scores = rewards[group_ood]
        if rollout_scores.mean() > ood_scores.mean():
            exit_mask |= group_ood
    return exit_mask

# ---- Wan OOD 训练脚本共享组件（原内嵌于各训练脚本，已去重）----


def gather_tensor(tensor, world_size):
    if world_size == 1:
        return tensor
    gather_list = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gather_list, tensor)
    return torch.cat(gather_list)


def set_seed(seed, device_specific=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device_specific and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + dist.get_rank() if dist.is_initialized() else seed)


class TextPromptDataset(Dataset):
    def __init__(self, dataset, split='train'):
        self.file_path = os.path.join(dataset, f'{split}.txt')
        with open(self.file_path, 'r') as f:
            self.prompts = [line.strip() for line in f.readlines()]

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx], "metadata": {'prompt': self.prompts[idx]}}

    @staticmethod
    def collate_fn(examples):
        prompts = [example["prompt"] for example in examples]
        metadatas = [example["metadata"] for example in examples]
        return prompts, metadatas


class GenevalPromptDataset(Dataset):
    def __init__(self, dataset, split='train'):
        self.file_path = os.path.join(dataset, f'{split}_metadata.jsonl')
        with open(self.file_path, 'r', encoding='utf-8') as f:
            self.metadatas = [json.loads(line) for line in f]
            self.prompts = [item['prompt'] for item in self.metadatas]

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx], "metadata": self.metadatas[idx]}

    @staticmethod
    def collate_fn(examples):
        prompts = [example["prompt"] for example in examples]
        metadatas = [example["metadata"] for example in examples]
        return prompts, metadatas


class OODPromptDataset(Dataset):
    """域外视频数据集。

    JSON 结构为 ``{prompt: [{video_path, reward, model_name}, ...]}``，
    每个 prompt 的条目数必须恒等于 ``n_ood_per_prompt``，条目已按 reward 降序排列。
    """

    def __init__(self, json_path, n_ood_per_prompt):
        with open(json_path, 'r', encoding='utf-8') as f:
            raw = json.load(f)
        count_dist = defaultdict(int)
        for entries in raw.values():
            count_dist[len(entries)] += 1
        assert set(count_dist.keys()) == {n_ood_per_prompt}, (
            f"OOD json 每个 prompt 的条目数必须恒为 {n_ood_per_prompt}，"
            f"实际分布 {dict(count_dist)}: {json_path}"
        )
        self.prompts = list(raw.keys())
        self.entries = [raw[prompt] for prompt in self.prompts]
        self.n_ood_per_prompt = n_ood_per_prompt

        missing = [
            entry["video_path"]
            for entries in self.entries
            for entry in entries
            if not os.path.exists(entry["video_path"])
        ]
        assert not missing, (
            f"OOD 视频缺失 {len(missing)} 个，前 5 个: {missing[:5]}"
        )

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        prompt = self.prompts[idx]
        return {
            "prompt": prompt,
            "metadata": {'prompt': prompt},
            "ood_entries": self.entries[idx],
        }


class DistributedKRepeatSampler(Sampler):
    def __init__(self, dataset, batch_size, k, num_replicas, rank, seed=0):
        self.dataset = dataset
        self.batch_size = batch_size
        self.k = k
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.total_samples = self.num_replicas * self.batch_size
        assert self.total_samples % self.k == 0, (
            f"k cannot divide n*b, k{k}-num_replicas{num_replicas}-batch_size{batch_size}"
        )
        self.m = self.total_samples // self.k
        self.epoch = 0

    def __iter__(self):
        while True:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g)[:self.m].tolist()
            repeated_indices = [idx for idx in indices for _ in range(self.k)]
            shuffled_indices = torch.randperm(len(repeated_indices), generator=g).tolist()
            shuffled_samples = [repeated_indices[i] for i in shuffled_indices]
            per_card_samples = []
            for i in range(self.num_replicas):
                start = i * self.batch_size
                end = start + self.batch_size
                per_card_samples.append(shuffled_samples[start:end])
            yield per_card_samples[self.rank]

    def set_epoch(self, epoch):
        self.epoch = epoch


def compute_text_embeddings(prompt, text_encoders, tokenizers, max_sequence_length, device):
    with torch.no_grad():
        prompt_embeds = encode_prompt(
            text_encoders, tokenizers, prompt, max_sequence_length
        )
        prompt_embeds = prompt_embeds.to(device)
    return prompt_embeds


def calculate_zero_std_ratio(prompts, gathered_rewards):
    prompt_array = np.array(prompts)
    unique_prompts, inverse_indices, counts = np.unique(
        prompt_array, return_inverse=True, return_counts=True
    )
    grouped_rewards = gathered_rewards['ori_avg'][np.argsort(inverse_indices)]
    split_indices = np.cumsum(counts)[:-1]
    reward_groups = np.split(grouped_rewards, split_indices)
    prompt_std_devs = np.array([np.std(group) for group in reward_groups])
    zero_std_count = np.count_nonzero(prompt_std_devs == 0)
    zero_std_ratio = zero_std_count / len(prompt_std_devs)
    return zero_std_ratio, prompt_std_devs.mean()


def get_transformer_layer_cls():
    return {WanTransformerBlock,}


def return_decay(step, decay_type):
    if decay_type == 0:
        flat, uprate, uphold = 0, 0.0, 0.0
    elif decay_type == 1:
        flat, uprate, uphold = 0, 0.001, 0.5
    elif decay_type == 2:
        flat, uprate, uphold = 75, 0.0075, 0.999
    else:
        raise ValueError(f"Unknown decay_type: {decay_type}")
    if step < flat:
        return 0.0
    return min((step - flat) * uprate, uphold)
