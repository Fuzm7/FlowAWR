import importlib.util
import os

base_path = os.path.join(os.path.dirname(__file__), "base.py")
spec = importlib.util.spec_from_file_location("base", base_path)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)


def get_config(name):
    return globals()[name]()


def _get_config(
    base_model="wan",
    model_size="1.3B",
    dataset="ocr",
    reward_fn=None,
    name="",
    method_name="flowawr",
):
    if reward_fn is None:
        reward_fn = {}

    config = base.get_config()
    config.base_model = base_model
    config.dataset = os.path.join(os.getcwd(), f"dataset/{dataset}")

    if model_size == "1.3B":
        config.pretrained.model = os.environ.get("WAN_MODEL", "Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
        config.height = 480
        config.width = 832
        config.frames = 81
        config.sample.guidance_scale = 4.5
        config.sample.train_batch_size = 4
        config.sample.test_batch_size = 8
    elif model_size == "14B":
        config.pretrained.model = os.environ.get("WAN_14B_MODEL", "Wan-AI/Wan2.1-T2V-14B-Diffusers")
        config.height = 480
        config.width = 832
        config.frames = 81
        config.sample.guidance_scale = 5.0
        config.sample.train_batch_size = 2
        config.sample.test_batch_size = 6

    config.use_fsdp = (model_size == "14B")

    config.fps = 16
    config.sample.num_steps = 20
    config.sample.eval_num_steps = 50
    config.mixed_precision = "bf16"

    config.sample.num_image_per_prompt = 8
    config.sample.sample_time_per_prompt = 1
    config.sample.num_batches_per_epoch = 1
    config.sample.global_std = False
    config.sample.kl_reward = 0.0

    config.train.batch_size = config.sample.train_batch_size
    # 非 OOD 路径按采样批次数推导；OOD 路径由 train_wan2_1_awr_ood.py 按训练侧
    # 每卡 clean latent 总量重新推导并覆盖本值。
    num_sample_batches = config.sample.num_batches_per_epoch * config.sample.sample_time_per_prompt
    config.train.gradient_accumulation_steps = num_sample_batches // 2 if num_sample_batches > 1 else 1
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.51
    config.train.learning_rate = 1e-4
    config.train.beta = 0.0001
    config.train.clip_range = 1e-3
    config.train.ema = True
    config.train.cfg = False

    config.decay_type = 2
    config.beta = 1.0
    config.train.adv_mode = "all"
    config.train.energy_mode = False
    config.train.hard_gating = False

    config.num_epochs = 100000
    config.save_freq = 20
    config.eval_freq = 20
    config.per_prompt_stat_tracking = True

    config.run_name = f"wan_{model_size}_{name}"
    config.method_name = method_name
    config.save_dir = f"{config.logdir}/wan/{name}/{model_size}/{method_name}"
    config.reward_fn = reward_fn
    config.prompt_fn = "general_ocr"
    config.use_dpo_reward = False
    config.dpo_lora_path = ""
    config.dpo_beta = 1.0
    config.dpo_reward_weight = 1.0
    config.train.flash_mode = False
    config.train.ultra_flash = False

    config.ood_enable = False
    config.ood_json = ""
    config.sample.ood_ratio = 0.0
    config.sample.n_ood_per_prompt = 0
    config.sample.ood_group_mode = "replace"
    config.sample.unique_prompts = 0
    config.train.ood_sft_mode = False
    config.train.ood_sft_weight = 1.0
    config.train.ood_exit_enable = False

    return config


def wan_awr_hpsv3():
    config = _get_config(
        model_size="1.3B",
        dataset="vidprom",
        reward_fn={"video_hpsv3": 1.0},
        name="hpsv3",
        method_name="VideoAWR",
    )
    config.train.energy_mode = True
    return config


def wan_awr_hpsv3_flash():
    config = _get_config(
        model_size="1.3B",
        dataset="vidprom",
        reward_fn={"video_hpsv3": 1.0},
        name="hpsv3_flash",
        method_name="VideoAWR",
    )
    config.train.energy_mode = True
    config.train.flash_mode = True
    return config


def wan_awr_videoalign():
    config = _get_config(
        model_size="1.3B",
        dataset="vidprom",
        reward_fn={"videoalign_score": 1.0},
        name="videoalign",
        method_name="VideoAWR",
    )
    config.train.energy_mode = True
    return config


def wan_awr_videoalign_flash():
    config = _get_config(
        model_size="1.3B",
        dataset="vidprom",
        reward_fn={"videoalign_score": 1.0},
        name="videoalign_flash",
        method_name="VideoAWR",
    )
    config.train.energy_mode = True
    config.train.flash_mode = True
    return config


def wan_awr_videoalign_14b():
    config = _get_config(
        model_size="14B",
        dataset="vidprom",
        reward_fn={"videoalign_score": 1.0},
        name="videoalign",
        method_name="VideoAWR",
    )
    config.train.energy_mode = True
    return config


def wan_awr_hpsv3_14b():
    config = _get_config(
        model_size="14B",
        dataset="vidprom",
        reward_fn={"video_hpsv3": 1.0},
        name="hpsv3",
        method_name="VideoAWR",
    )
    config.train.energy_mode = True
    return config


def wan_awr_hpsv3_flash_14b():
    config = _get_config(
        model_size="14B",
        dataset="vidprom",
        reward_fn={"video_hpsv3": 1.0},
        name="hpsv3_flash",
        method_name="VideoAWR",
    )
    config.train.energy_mode = True
    config.train.flash_mode = True
    return config


def wan_awr_videoalign_flash_14b():
    config = _get_config(
        model_size="14B",
        dataset="vidprom",
        reward_fn={"videoalign_score": 1.0},
        name="videoalign_flash",
        method_name="VideoAWR",
    )
    config.train.energy_mode = True
    config.train.flash_mode = True
    return config


def wan_awr_hpsv3_ultraflash():
    config = wan_awr_hpsv3_flash()
    config.train.ultra_flash = True
    return config


def wan_awr_videoalign_ultraflash():
    config = wan_awr_videoalign_flash()
    config.train.ultra_flash = True
    return config


def wan_awr_hpsv3_ultraflash_14b():
    config = wan_awr_hpsv3_flash_14b()
    config.train.ultra_flash = True
    return config


def wan_awr_videoalign_ultraflash_14b():
    config = wan_awr_videoalign_flash_14b()
    config.train.ultra_flash = True
    return config


def _apply_wan_ood_overrides(config, n_ood_per_prompt=4, ood_ratio=0.5):
    """启用域外混合训练。

    默认口径为 replace 分组 + ood_sft_mode（域外样本走纯速度回归、不参与组内
    softmax 归一化）+ 关闭 ood_exit。``ood_group_mode`` / ``ood_sft_mode`` /
    ``ood_exit_enable`` / ``n_ood_per_prompt`` / ``ood_ratio`` / ``ood_json``
    均可由启动脚本以 ``--config.*`` 覆盖，故此处不再按开关派生多个入口函数。
    """
    config.ood_enable = True
    config.sample.n_ood_per_prompt = n_ood_per_prompt
    config.sample.ood_ratio = ood_ratio
    config.sample.ood_group_mode = "replace"
    config.train.ood_sft_mode = True
    config.train.ood_exit_enable = False
    config.run_name = f"{config.run_name}_ood"
    config.save_dir = f"{config.save_dir}_ood"
    return config


def wan_awr_hpsv3_flash_ood():
    return _apply_wan_ood_overrides(wan_awr_hpsv3_flash())


def wan_awr_hpsv3_ultraflash_ood():
    return _apply_wan_ood_overrides(wan_awr_hpsv3_ultraflash())
