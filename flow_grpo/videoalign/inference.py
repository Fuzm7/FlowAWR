import ast
import json
import os
import pdb
from collections.abc import Mapping
import pandas as pd

import torch
from vision_process import process_vision_info

from data import DataConfig
from utils import ModelConfig, PEFTLoraConfig, TrainingConfig
from utils import load_model_from_checkpoint
from train_reward import create_model_and_processor
from prompt_template import build_prompt

# pip install trl

def load_configs_from_json(config_path):
    with open(config_path, "r") as f:
        config_dict = json.load(f)

    # del config_dict["training_args"]["_n_gpu"]
    del config_dict["data_config"]["meta_data"]
    del config_dict["data_config"]["data_dir"]

    return config_dict["data_config"], None, config_dict["model_config"], config_dict["peft_lora_config"], \
           config_dict["inference_config"] if "inference_config" in config_dict else None

class VideoVLMRewardInference():
    def __init__(self, load_from_pretrained, load_from_pretrained_step=-1, device='cuda', dtype=torch.bfloat16, model_name_or_path=None):
        config_path = os.path.join(load_from_pretrained, "model_config.json")
        data_config, _, model_config, peft_lora_config, inference_config = load_configs_from_json(config_path)
        data_config = DataConfig(**data_config)
        model_config = ModelConfig(**model_config)
        if model_name_or_path is not None:
            model_config.model_name_or_path = model_name_or_path
        peft_lora_config = PEFTLoraConfig(**peft_lora_config)

        training_args = TrainingConfig(
            load_from_pretrained=load_from_pretrained,
            load_from_pretrained_step=load_from_pretrained_step,
            gradient_checkpointing=False,
            disable_flash_attn2=False,
            bf16=True if dtype == torch.bfloat16 else False,
            fp16=True if dtype == torch.float16 else False,
            output_dir="",
        )
        
        model, processor, peft_config = create_model_and_processor(
            model_config=model_config,
            peft_lora_config=peft_lora_config,
            training_args=training_args,
        )

        self.device = device

        model, checkpoint_step = load_model_from_checkpoint(model, load_from_pretrained, load_from_pretrained_step)
        model.eval()

        self.model = model
        self.processor = processor

        self.model.to(self.device)

        self.data_config = data_config

        self.inference_config = inference_config

    def _norm(self, reward):
        if self.inference_config is None:
            return reward
        else:
            reward['VQ'] = (reward['VQ'] - self.inference_config['VQ_mean']) / self.inference_config['VQ_std']
            reward['MQ'] = (reward['MQ'] - self.inference_config['MQ_mean']) / self.inference_config['MQ_std']
            reward['TA'] = (reward['TA'] - self.inference_config['TA_mean']) / self.inference_config['TA_std']
            return reward

    def _pad_sequence(self, sequences, attention_mask, max_len, padding_side='right'):
        """
        Pad the sequences to the maximum length.
        """
        assert padding_side in ['right', 'left']
        if sequences.shape[1] >= max_len:
            return sequences, attention_mask
        
        pad_len = max_len - sequences.shape[1]
        padding = (0, pad_len) if padding_side == 'right' else (pad_len, 0)

        sequences_padded = torch.nn.functional.pad(sequences, padding, 'constant', self.processor.tokenizer.pad_token_id)
        attention_mask_padded = torch.nn.functional.pad(attention_mask, padding, 'constant', 0)

        return sequences_padded, attention_mask_padded
    
    def _prepare_input(self, data):
        """
        Prepare `inputs` before feeding them to the model, converting them to tensors if they are not already and
        handling potential state.
        """
        if isinstance(data, Mapping):
            return type(data)({k: self._prepare_input(v) for k, v in data.items()})
        elif isinstance(data, (tuple, list)):
            return type(data)(self._prepare_input(v) for v in data)
        elif isinstance(data, torch.Tensor):
            kwargs = {"device": self.device}
            ## TODO: Maybe need to add dtype
            # if self.is_deepspeed_enabled and (torch.is_floating_point(data) or torch.is_complex(data)):
            #     # NLP models inputs are int/uint and those get adjusted to the right dtype of the
            #     # embedding. Other models such as wav2vec2's inputs are already float and thus
            #     # may need special handling to match the dtypes of the model
            #     kwargs.update({"dtype": self.accelerator.state.deepspeed_plugin.hf_ds_config.dtype()})
            return data.to(**kwargs)
        return data
    
    def _prepare_inputs(self, inputs):
        """
        Prepare `inputs` before feeding them to the model, converting them to tensors if they are not already and
        handling potential state.
        """
        inputs = self._prepare_input(inputs)
        if len(inputs) == 0:
            raise ValueError
        return inputs
    
    def prepare_batch(self, video_paths, prompts, fps=None, num_frames=None, max_pixels=None,):
        fps = self.data_config.fps if fps is None else fps
        num_frames = self.data_config.num_frames if num_frames is None else num_frames
        max_pixels = self.data_config.max_frame_pixels if max_pixels is None else max_pixels

        if num_frames is None:
            chat_data = [
                [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "video", 
                                "video": f"file://{video_path}", 
                                "max_pixels": max_pixels, 
                                "fps": fps,
                                "sample_type": self.data_config.sample_type,
                            },
                            {"type": "text", "text": build_prompt(prompt, self.data_config.eval_dim, self.data_config.prompt_template_type)},
                        ],
                    },
                ] for video_path, prompt in zip(video_paths, prompts)
            ]
        else:
            chat_data = [
                [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "video",
                                "video": f"file://{video_path}", 
                                "max_pixels": max_pixels, 
                                "nframes": num_frames,
                                "sample_type": self.data_config.sample_type,
                            },
                            {"type": "text", "text": build_prompt(prompt, self.data_config.eval_dim, self.data_config.prompt_template_type)},
                        ],
                    },
                ] for video_path, prompt in zip(video_paths, prompts)
            ]
        image_inputs, video_inputs = process_vision_info(chat_data)

        # import torchvision.transforms as T
        # if video_inputs is not None:
        #     grayscale = T.Grayscale(num_output_channels=3)
        #     video_inputs = [grayscale(v).to(torch.uint8) for v in video_inputs]

        batch = self.processor(
            text=self.processor.apply_chat_template(chat_data, tokenize=False, add_generation_prompt=True),
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            videos_kwargs={"do_rescale": True},
        )
        batch = self._prepare_inputs(batch)
        return batch   

    def reward(self, video_paths, prompts, fps=None, num_frames=None, max_pixels=None, use_norm=True):
        """
        Inputs:
            video_paths: List[str], B paths of the videos.
            prompts: List[str], B prompts for the videos.
            eval_dims: List[str], N evaluation dimensions.
            fps: float, sample rate of the videos. If None, use the default value in the config.
            num_frames: int, number of frames of the videos. If None, use the default value in the config.
            max_pixels: int, maximum pixels of the videos. If None, use the default value in the config.
            use_norm: bool, whether to rescale the output rewards
        Outputs:
            Rewards: List[dict], N + 1 rewards of the B videos.
        """
        assert fps is None or num_frames is None, "fps and num_frames cannot be set at the same time."
        batch = self.prepare_batch(video_paths, prompts, fps, num_frames, max_pixels)
        rewards = self.model(
            return_dict=True,
            **batch
        )["logits"]

        rewards = [{'VQ': reward[0].item(), 'MQ': reward[1].item(), 'TA': reward[2].item()} for reward in rewards]
        for i in range(len(rewards)):
            if use_norm:
                rewards[i] = self._norm(rewards[i])
            rewards[i]['Overall'] = rewards[i]['VQ'] + rewards[i]['MQ'] + rewards[i]['TA']

        return rewards


if __name__ == "__main__":
    load_from_pretrained = "/path/to/VideoReward"
    device = "cuda:0"
    dtype = torch.bfloat16

    inferencer = VideoVLMRewardInference(load_from_pretrained, device=device, dtype=dtype)

    video_paths = [
        "/path/to/video_1.mp4",
        "/path/to/video_2.mp4",
    ]

    prompts = [
        "A basketball player, dressed in a sleek black jersey with white accents and black shorts, dribbles past an opponent with a series of quick, controlled movements, his hands moving rhythmically as he navigates through the defensive player. The ball bounces with a consistent rhythm, reflecting the player's focus and determination. As he reaches the basket, he leaps into the air, his legs extending fully, and releases the ball with a powerful shot, the ball soaring through the air in a smooth arc towards the hoop. The moment the ball touches the rim, it swishes through the net, and the player's face lights up with joy as he celebrates the score, raising his arms high in triumph. The shot is captured in a single, focused motion, emphasizing the player's skill and the moment of success. The background is a blur of motion, highlighting the intensity of the play, while the lighting illuminates the player's face and the ball's trajectory, creating a dynamic and exciting visual. The scene is framed with a wide-angle lens, capturing the player's movement and the surrounding court, with the basketball hoop and the player's celebration as the central focus.",
        "A basketball player, dressed in a sleek black jersey with white accents and black shorts, dribbles past an opponent with a series of quick, controlled movements, his hands moving rhythmically as he navigates through the defensive player. The ball bounces with a consistent rhythm, reflecting the player's focus and determination. As he reaches the basket, he leaps into the air, his legs extending fully, and releases the ball with a powerful shot, the ball soaring through the air in a smooth arc towards the hoop. The moment the ball touches the rim, it swishes through the net, and the player's face lights up with joy as he celebrates the score, raising his arms high in triumph. The shot is captured in a single, focused motion, emphasizing the player's skill and the moment of success. The background is a blur of motion, highlighting the intensity of the play, while the lighting illuminates the player's face and the ball's trajectory, creating a dynamic and exciting visual. The scene is framed with a wide-angle lens, capturing the player's movement and the surrounding court, with the basketball hoop and the player's celebration as the central focus.",
        # "In a bright, sunny outdoor setting, a person is captured from the waist down, skateboarding on a wide, paved asphalt path. The individual is wearing a black jacket, dark grey jeans with the cuffs rolled up to reveal white socks, and brown and black sneakers with white soles. The skateboard has a grey grip tape surface, a red underside, and light-colored wheels. The path is lined on both sides with rows of young, leafless trees, modern streetlights, and low concrete curbs. In the distant background, several large, white apartment buildings stand against a clear blue sky. At the beginning of the video, the person is riding the skateboard with their knees bent. Then, they begin to prepare for a trick by lowering their body and bending their knees further. Subsequently, they crouch down, place their hands on their thighs, and then jump upwards. As they jump, their feet kick the edge of the skateboard, causing it to flip in the air. During the jump, the person's upper body and head, wearing glasses, become partially visible. After the board completes its rotation, the person lands back on it with both feet, bending their knees to absorb the impact and continue riding. Throughout the entire process, the background remains consistent, with the bright sunlight casting a clear shadow of the person and the skateboard on the ground.",
        # """In an indoor basketball gymnasium with a polished wooden floor and grey walls, two men are engaged in a one-on-one basketball drill. On the left, a man with his back to the camera wears a blue sleeveless jersey, black shorts with white side panels, white socks, and black-and-white sneakers. On the right, a man with dark hair and a beard, wearing a red and white sleeveless jersey and matching shorts, white socks, and red sneakers, is in a low athletic stance. He begins to dribble an orange basketball with his right hand. As he dribbles, the defender in blue maintains his defensive posture. The player in red then performs a crossover dribble and drives past the defender to the right. The camera follows his movement, and in the background, wooden bleachers and a door with a white sign reading "101" become visible. As the player in red approaches the basket, he gathers the ball and jumps for a layup. The camera angle shifts to show the basketball hoop, which has an orange rim, a white net, and a transparent backboard with a "Spalding" logo. Two other players in blue jerseys are now visible in the background, watching. The basketball successfully goes through the net. After landing, the player in red turns towards the camera, raises both arms with clenched fists, opens his mouth in a triumphant shout, and then bounces the ball with his left hand while maintaining his celebratory pose. The player in blue stands to his right, with his back to the camera. In the background, a green emergency exit sign can be seen on the wall.""",
    ]


    # 视频初始画面为足球场低角度视角，前景是绿色草坪与白色标线，标线旁有一颗白红相间的足球。一名身着蓝色上衣、白色带蓝条纹短裤、蓝色长袜（袜身蓝白相间）及黑（右脚）、白（左脚）足球鞋的球员从画面右侧向足球移动。背景中，黄白框架的足球门、两侧看台、远处树木、灯柱与蓝天清晰可见。球员先小步跑动接近足球，随后在球前停下并转向球门，右腿后摆、左腿支撑，右脚发力将球踢向球门；足球飞向球门时，球员继续向前跑动，背景里球门远处还有身着红色与白色球衣的球员活动。
    # In an indoor basketball gymnasium with a polished wooden floor and grey walls, two men are engaged in a one-on-one basketball drill. On the left, a man with his back to the camera wears a blue sleeveless jersey, black shorts with white side panels, white socks, and black-and-white sneakers. On the right, a man with dark hair and a beard, wearing a red and white sleeveless jersey and matching shorts, white socks, and red sneakers, is in a low athletic stance. He begins to dribble an orange basketball with his right hand. As he dribbles, the defender in blue maintains his defensive posture. The player in red then performs a crossover dribble and drives past the defender to the right. The camera follows his movement, and in the background, wooden bleachers and a door with a white sign reading "101" become visible. As the player in red approaches the basket, he gathers the ball and jumps for a layup. The camera angle shifts to show the basketball hoop, which has an orange rim, a white net, and a transparent backboard with a "Spalding" logo. Two other players in blue jerseys are now visible in the background, watching. The basketball successfully goes through the net. After landing, the player in red turns towards the camera, raises both arms with clenched fists, opens his mouth in a triumphant shout, and then bounces the ball with his left hand while maintaining his celebratory pose. The player in blue stands to his right, with his back to the camera. In the background, a green emergency exit sign can be seen on the wall.
    # 视频中，一名身着红色运动短裤、红色运动上衣与红白色运动鞋的跑步者，在蓝色跑道上持续奔跑。画面聚焦于跑步者的下半身及部分上半身，其动作呈循环状态：左腿向前迈出、右腿向后摆动，同时双臂自然地前后摆动以配合跑步节奏。背景里，蓝色跑道分布着白色与黄色标线，跑道外侧是绿色草地，远处可见空旷的白色看台、树木及高耸的灯柱，天空为晴朗的蓝色。整个过程中，跑步者衣物随动作产生动态褶皱，肌肉随奔跑规律收缩舒张，背景环境始终静态，无其他人物或物体出现，仅跑步者的奔跑动作持续循环推进。
    # The video captures an intense table tennis match inside a well-lit indoor sports hall with a polished wooden floor. In the foreground, a young East Asian man with short black hair is positioned on the left side of a blue "ONUSTY" table tennis table. He is wearing a white polo shirt with logos on the sleeve and chest, dark blue shorts with a white stripe, and white socks. Holding a red paddle in his right hand, he is in a low, focused stance, preparing to return the ball. In the background, rows of empty beige and purple spectator seats are visible, along with blue banners featuring the "Dipoloy" logo. A large electronic scoreboard on the wall displays "TEAM 4 & 5 17:47:20" and lists player names such as "SHARIF," "SAM WEI HONG," and "SAM WEI JIE," with a score of "15 7 3" and "15 10 4." As the rally begins, the white ball flies towards him, and he swiftly hits it with a forehand stroke. The camera then pans to the right, following the ball's trajectory. On the opposite side of the table, another young East Asian man with light brown hair, dressed in a black t-shirt, black shorts, and white Adidas sneakers, prepares to receive the shot. He also holds a red paddle and lunges to his right to hit the ball. Throughout the scene, the scoreboard and banners with text like "X1booc," "NUS," and "STUDENT CHAMPIONSHIP" remain visible in the background, where a few spectators can also be seen sitting.
    # 视频初始画面聚焦于健身房内的前景人物下半身，该人物穿着黑色紧身裤与黑灰配色运动鞋，左小腿处有纹身；背景中，大镜子反射出另一位身着红色上衣、黑色紧身裤和红色运动鞋的人物。随后镜头向上移动，呈现前景人物的上半身——其身穿灰色短袖，双手各握持一个带有红色标识文字的黑色哑铃，正执行哑铃弯举动作：手臂从自然下垂位置向上弯曲抬起，哑铃随之上移至肩部附近，再缓慢下放回初始位置，动作循环往复。过程中，前景人物的手臂肌肉随动作收缩、舒张，线条分明。背景里，健身房的各类器械（如哑铃架、训练椅）有序排列，大面积镜子不仅反射出室内陈设，也映出窗外光线与天花板悬挂的灯具，整体空间因自然光与人工照明显得明亮开阔。
    # In a bright, sunny outdoor setting, a person is captured from the waist down, skateboarding on a wide, paved asphalt path. The individual is wearing a black jacket, dark grey jeans with the cuffs rolled up to reveal white socks, and brown and black sneakers with white soles. The skateboard has a grey grip tape surface, a red underside, and light-colored wheels. The path is lined on both sides with rows of young, leafless trees, modern streetlights, and low concrete curbs. In the distant background, several large, white apartment buildings stand against a clear blue sky. At the beginning of the video, the person is riding the skateboard with their knees bent. Then, they begin to prepare for a trick by lowering their body and bending their knees further. Subsequently, they crouch down, place their hands on their thighs, and then jump upwards. As they jump, their feet kick the edge of the skateboard, causing it to flip in the air. During the jump, the person's upper body and head, wearing glasses, become partially visible. After the board completes its rotation, the person lands back on it with both feet, bending their knees to absorb the impact and continue riding. Throughout the entire process, the background remains consistent, with the bright sunlight casting a clear shadow of the person and the skateboard on the ground.
    # 视频开始时，一名身穿红色无袖运动背心、黑色短裤、荧光绿色运动鞋的男子在田径场的起跑器上呈起跑姿势，身体前倾、肌肉紧绷。随后他蹬离起跑器，沿黑色跑道快速助跑，双臂有力摆动，双腿交替迈开，背景中可见空旷的浅色看台、绿色草地及高大的照明灯柱。助跑过程中，他接近跳远区域后起跳，身体腾空并越过白色的跨栏架（跨栏架旁有橙色锥桶），随后落入沙坑，沙子因冲击飞溅。落地后，他先弯腰在沙坑中调整，随后站起并侧身观察沙坑内的落点，背景始终是晴朗天空下的田径场设施，包括远处的树木与看台建筑。
    # In a brightly lit indoor setting, a chef dressed in a white long-sleeved chef's coat and a beige apron is preparing salmon sushi on a polished brown wooden countertop. On the countertop, there is a brown and green speckled ceramic bowl filled with white sushi rice on the left, and a large piece of raw salmon fillet with distinct orange and white marbling next to it. To the right, there is a wooden sushi rolling mat, a yellow and white flower-shaped decoration, and a small bowl containing white sesame seeds. The chef, whose hands and forearms are visible, holds a silver knife with a black handle in their right hand. With their left hand, they steady a piece of salmon nigiri sushi that has already been cut into two. Throughout the video, the chef repeatedly makes precise cuts on the sushi, separating it into smaller pieces. During this process, a small white saucer with dark soy sauce appears on the left side of the frame. The background is a plain, light gray wall, and the scene is illuminated by soft light coming from the left, creating a focused and professional atmosphere. The chef's white coat has ruffled cuffs and a decorative white fabric detail on the left shoulder.


    with torch.no_grad():
        rewards = inferencer.reward(video_paths, prompts, use_norm=True)
        print(rewards[0]['TA'])
        print(rewards[1]['TA'])
        # print(rewards[2]['VQ'])



