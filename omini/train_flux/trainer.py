import lightning as L
from diffusers.pipelines import FluxPipeline
import torch
import wandb
import os
import yaml
from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict
from torch.utils.data import DataLoader
from safetensors.torch import load_file,save_file
import time
from torch.distributions import Normal
from diffusers.models.embeddings import (
    Timesteps,          # 时间步基础编码（sin/cos 位置编码）
    TimestepEmbedding,# 时间步嵌入层（MLP 映射）
    PixArtAlphaTextProjection
)
from PIL import Image, ImageFilter
from typing import List, Optional, Dict, Tuple
import torch.nn as nn
import prodigyopt
from accelerate import load_checkpoint_and_dispatch

from ..pipeline.flux_omini import transformer_forward, encode_images, Condition


# LOCAL_FLUX_DIR = "/FLUX.1-dev"
def get_rank():
    try:
        rank = int(os.environ.get("LOCAL_RANK"))
    except:
        rank = 0
    return rank


def get_config():
    config_path = os.environ.get("OMINI_CONFIG")
    assert config_path is not None, "Please set the OMINI_CONFIG environment variable"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def init_wandb(wandb_config, run_name):
    import wandb

    try:
        assert os.environ.get("WANDB_API_KEY") is not None
        wandb.init(
            project=wandb_config["project"],
            name=run_name,
            config={},
        )
    except Exception as e:
        print("Failed to initialize WanDB:", e)
class time_text_embed_module2(nn.Module):
    def __init__(self, embedding_dim, pooled_projection_dim):
        super().__init__()

        # 1. 时间步 t 的基础编码（保持不变）
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)

        # 2. 时间间隔 Δt 的基础编码（保持不变，与 t 结构一致）
        self.delta_time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.delta_timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)

        # 5. 原有文本特征编码（保持不变）
        self.text_embedder = PixArtAlphaTextProjection(pooled_projection_dim, embedding_dim, act_fn="silu")

    def forward(self, timestep, delta_timestep, pooled_projection):
        """
        输入扩展：新增 delta_timestep（Δt = t - r）
        args:
            timestep: 时间步 t（shape: (batch_size,)）
            delta_timestep: 时间间隔 Δt = t - r（shape: (batch_size,)）
            pooled_projection: 文本池化特征（shape: (batch_size, pooled_projection_dim)）
        return:
            conditioning: 融合 (t, Δt, 文本) 的最终嵌入（shape: (batch_size, embedding_dim)）
        """
        # 步骤1：对 t 进行位置编码（原有逻辑不变）
        t_proj = self.time_proj(timestep)
        t_emb = self.timestep_embedder(t_proj.to(dtype=pooled_projection.dtype))  # (B, embedding_dim)

        # 步骤2：对 Δt 进行位置编码（原有逻辑不变）
        delta_t_proj = self.delta_time_proj(delta_timestep)
        delta_t_emb = self.delta_timestep_embedder(delta_t_proj.to(dtype=pooled_projection.dtype))  # (B, embedding_dim)

        # 修改后的代码：强制将输入移动到模型层所在的设备
        target_device = self.text_embedder.linear_1.weight.device  # 获取模型层的设备(通常是cuda:0)
        text_emb = self.text_embedder(pooled_projection.to(target_device))
        # 步骤6：文本特征编码（原有逻辑不变）
        # text_emb = self.text_embedder(pooled_projection)  # (B, embedding_dim)

        # 步骤7：融合时间特征和文本特征（保持原有加法融合逻辑）
        conditioning = t_emb + delta_t_emb + text_emb

        return conditioning
class time_text_embed_module1(nn.Module):
    def __init__(self, embedding_dim, pooled_projection_dim):
        super().__init__()

        # 1. 时间步 t 的基础编码（保持不变）
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)


        # 3. 为 t 单独创建 2-layer MLP（结构与原共享MLP一致）
        self.timestep_mlp = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),  # 输入维度为 embedding_dim（单一时间编码维度）
            nn.SiLU(),  # 激活函数保持与原代码一致
            nn.Linear(embedding_dim, embedding_dim)
        )

        # 4. 为 Δt 单独创建 2-layer MLP（与 t 的MLP结构完全一致，保证对等处理）
        self.delta_timestep_mlp = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )

        # 5. 原有文本特征编码（保持不变）
        self.text_embedder = PixArtAlphaTextProjection(pooled_projection_dim, embedding_dim, act_fn="silu")

    def forward(self, timestep, delta_timestep, pooled_projection):
        """
        输入扩展：新增 delta_timestep（Δt = t - r）
        args:
            timestep: 时间步 t（shape: (batch_size,)）
            delta_timestep: 时间间隔 Δt = t - r（shape: (batch_size,)）
            pooled_projection: 文本池化特征（shape: (batch_size, pooled_projection_dim)）
        return:
            conditioning: 融合 (t, Δt, 文本) 的最终嵌入（shape: (batch_size, embedding_dim)）
        """
        # 步骤1：对 t 进行位置编码（原有逻辑不变）
        t_proj = self.time_proj(timestep)
        t_emb = self.timestep_embedder(t_proj.to(dtype=pooled_projection.dtype))  # (B, embedding_dim)

        # 步骤2：对 Δt 进行位置编码（原有逻辑不变）
        delta_t_proj = self.time_proj(delta_timestep)
        delta_t_emb = self.timestep_embedder(delta_t_proj.to(dtype=pooled_projection.dtype))  # (B, embedding_dim)

        # 步骤3：t 经过独立的 2-layer MLP 处理
        t_emb_processed = self.timestep_mlp(t_emb)  # (B, embedding_dim)

        # 步骤4：Δt 经过独立的 2-layer MLP 处理
        delta_t_emb_processed = self.delta_timestep_mlp(delta_t_emb)  # (B, embedding_dim)

        # 步骤5：求和融合 t 和 Δt 的处理结果（符合你的要求）
        fused_time_emb = t_emb_processed + delta_t_emb_processed  # (B, embedding_dim)

        # 步骤6：文本特征编码（原有逻辑不变）
        text_emb = self.text_embedder(pooled_projection)  # (B, embedding_dim)

        # 步骤7：融合时间特征和文本特征（保持原有加法融合逻辑）
        conditioning = fused_time_emb + text_emb

        return conditioning
class time_text_embed_module(nn.Module):
    def __init__(self, embedding_dim, pooled_projection_dim):
        super().__init__()

        # 1. 原有时间步基础编码（保持不变）
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)

        # 2. 新增 delta_t（Δt = t - r）的基础编码（与 t 结构一致）
        self.delta_time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.delta_timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)

        # 3. 融合 t 和 delta_t 编码的 2-layer MLP（论文 4.3 要求）
        self.time_fusion_mlp = nn.Sequential(
            nn.Linear(2 * embedding_dim, embedding_dim),
            nn.SiLU(),  # 激活函数与文本嵌入保持一致
            nn.Linear(embedding_dim, embedding_dim)
        )

        # 4. 原有文本特征编码（保持不变）
        self.text_embedder = PixArtAlphaTextProjection(pooled_projection_dim, embedding_dim, act_fn="silu")

    def forward(self, timestep, delta_timestep, pooled_projection):
        """
        输入扩展：新增 delta_timestep（Δt = t - r）
        args:
            timestep: 时间步 t（shape: (batch_size,)）
            delta_timestep: 时间间隔 Δt = t - r（shape: (batch_size,)）
            pooled_projection: 文本池化特征（shape: (batch_size, pooled_projection_dim)）
        return:
            conditioning: 融合 (t, Δt, 文本) 的最终嵌入（shape: (batch_size, embedding_dim)）
        """
        # 步骤1：对 t 进行编码（原有逻辑）
        t_proj = self.time_proj(timestep)
        t_emb = self.timestep_embedder(t_proj.to(dtype=pooled_projection.dtype))  # (B, pos_embed_dim)

        # 步骤2：对 Δt 进行编码（新增逻辑，与 t 编码结构完全一致）
        delta_t_proj = self.delta_time_proj(delta_timestep)
        delta_t_emb = self.delta_timestep_embedder(delta_t_proj.to(dtype=pooled_projection.dtype))  # (B, pos_embed_dim)

        # 步骤3：融合 t 和 Δt 的编码（论文核心要求：uθ(·, r, t) ≜ net(·, t, t−r)）
        combined_time_emb = torch.cat([t_emb, delta_t_emb], dim=-1)  # (B, 2*pos_embed_dim)
        fused_time_emb = self.time_fusion_mlp(combined_time_emb)  # (B, embedding_dim)

        # 步骤4：文本特征编码（原有逻辑不变）
        text_emb = self.text_embedder(pooled_projection)  # (B, embedding_dim)

        # 步骤5：融合时间特征和文本特征（加法融合，保持原有逻辑）
        conditioning = fused_time_emb + text_emb

        return conditioning

class OminiModel(L.LightningModule):
    def __init__(
        self,
        flux_pipe_id: str,
        lora_path: str = None,
        lora_config: dict = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        model_config: dict = {},
        adapter_names: List[str] = [None, None, "default"],
        optimizer_config: dict = None,
        gradient_checkpointing: bool = False,
        time_layers_path : str = None,
        omega_: float = 2,
        kappa: float = 0.9,
    ):
        # Initialize the LightningModule
        super().__init__()
        self.model_config = model_config
        self.optimizer_config = optimizer_config
        # Load the Flux pipeline
        self.flux_pipe: FluxPipeline = FluxPipeline.from_pretrained(
            flux_pipe_id, torch_dtype=dtype
        ).to(device)

        # self.flux_pipe: FluxPipeline = FluxPipeline.from_pretrained(
        #     LOCAL_FLUX_DIR,  # 关键：替换为本地目录路径
        #     torch_dtype=dtype,
        #     local_files_only=True,
        #     device_map="balanced",
        #     max_memory={0: "12GB", 1: "12GB"},
        # )
        self.transformer = self.flux_pipe.transformer
        # print(f"dtype:{dtype},self.dtype:{self.dtype},{self.flux_pipe.dtype}")
        # flux_pipe  = FluxPipeline.from_pretrained(
        #     LOCAL_FLUX_DIR, torch_dtype=dtype, local_files_only=True,use_auth_token=False
        # ).to(device)
        # print(f"type:{dtype}")
        # target_num_layers = 1  # 原默认 19，自定义修改
        # target_num_single_layers = 1  # 原默认 38，自定义修改
        #
        # # 第三步：提取现有 Transformer 的配置（复用所有其他参数，仅修改两个目标参数）
        # # 第三步：提取现有 Transformer 的配置（复用所有其他参数，仅修改两个目标参数）
        # original_transformer = flux_pipe.transformer
        # original_state_dict = original_transformer.state_dict()
        # transformer_config = {
        #     "patch_size": 1,  # Flux 固定为 1
        #     "in_channels": 64,  # Flux 固定为 64
        #     "out_channels": original_transformer.out_channels,  # 非配置属性，直接访问（无警告）
        #     "num_layers": target_num_layers,  # 替换为自定义值
        #     "num_single_layers": target_num_single_layers,  # 替换为自定义值
        #     # 以下所有配置属性，均改为通过 .config 访问（消除弃用警告）
        #     "attention_head_dim": original_transformer.config.attention_head_dim,
        #     "num_attention_heads": original_transformer.config.num_attention_heads,
        #     "joint_attention_dim": original_transformer.config.joint_attention_dim,
        #     "pooled_projection_dim": original_transformer.config.pooled_projection_dim,
        #     "guidance_embeds": hasattr(original_transformer.config,
        #                                "guidance_embeds") and original_transformer.config.guidance_embeds,
        #     "axes_dims_rope": original_transformer.pos_embed.axes_dim  # 非配置属性，直接访问（无警告）
        # }
        #
        # # 第四步：重新实例化 Transformer（使用修改后的配置）
        # # 注意：这里需要导入 Flux Transformer 的实际类（通常是 FluxTransformer）
        # from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
        # new_transformer = FluxTransformer2DModel(**transformer_config).to(device, dtype=dtype)
        #
        # print("过滤预训练权重，仅保留 1 层对应权重...")
        # filtered_state_dict = {}
        # for key, value in original_state_dict.items():
        #     # 情况 1：非 transformer_blocks 相关的通用权重（全部保留，如嵌入层、位置编码等）
        #     if not key.startswith("transformer_blocks."):
        #         filtered_state_dict[key] = value
        #     # 情况 2：transformer_blocks 相关的权重，仅保留第 0 层（对应 target_num_layers=1）
        #     else:
        #         # 提取层索引（如 "transformer_blocks.0.attention.q_proj.weight" 中的 0）
        #         layer_index = int(key.split(".")[1])
        #         if layer_index == 0:  # 只保留第 0 层（即第 1 层，对应 target_num_layers=1）
        #             filtered_state_dict[key] = value
        #
        # # 步骤 5：加载过滤后的权重（strict=False 忽略新模型中不存在的多余层权重）
        # print("加载过滤后的 1 层权重...")
        # missing_keys, unexpected_keys = new_transformer.load_state_dict(
        #     filtered_state_dict,
        #     strict=False
        # )
        #
        # # 打印加载日志（验证是否只加载了 1 层权重）
        # print(f"\n权重加载完成：")
        # print(f"  缺失键（新模型有、预训练无，正常）：{len(missing_keys)} 个")
        # print(f"  多余键（预训练有、新模型无，已过滤，正常）：{len(unexpected_keys)} 个")
        # if len(missing_keys) > 0 and missing_keys[:5]:
        #     print(f"  缺失键示例：{missing_keys[:5]}")
        # if len(unexpected_keys) > 0 and unexpected_keys[:5]:
        #     print(f"  多余键示例：{unexpected_keys[:5]}")
        #
        # # 步骤 6：将 1 层 Transformer 移到 GPU（仅这一步占用 GPU 显存）
        # print("\n将 1 层 Transformer 移到 GPU...")
        # new_transformer = new_transformer.to(device, dtype=dtype)
        #
        # # 步骤 7：构建轻量管道（仅保留必要组件，其他移到 CPU）
        # print("构建轻量 FluxPipeline...")
        # self.flux_pipe = flux_pipe
        # self.flux_pipe.transformer = new_transformer # 替换为 1 层 Transformer
        # self.transformer = new_transformer
        #
        # # 步骤 8：验证显存占用和模型层数
        # print(f"\n最终验证：")
        # print(f"  新模型 transformer_blocks 长度：{len(self.transformer.transformer_blocks)}")

        # self.transformer = self.flux_pipe.transformer
        self.transformer.gradient_checkpointing = gradient_checkpointing
        self.transformer.train()

        # Freeze the Flux pipeline
        self.flux_pipe.text_encoder.requires_grad_(False).eval()
        self.flux_pipe.text_encoder_2.requires_grad_(False).eval()
        self.flux_pipe.vae.requires_grad_(False).eval()
        self.adapter_names = adapter_names
        self.adapter_set = set([each for each in adapter_names if each is not None])

        # Initialize LoRA layers
        self.time_layers= self.replace_and_freeze_time_text_embed(self.flux_pipe, time_layers_path)

        self.lora_layers = self.init_lora(lora_path, lora_config, device)
        self.omega = omega_*(1-kappa)
        self.kappa = kappa
        # devices = set()
        # for name, param in self.transformer.named_parameters():
        #     devices.add(str(param.device))
        #
        # print(f"模型分布在: {devices}")

    # def on_before_optimizer_step(self, optimizer):
    #     # 这个函数会在 optimizer.step() 之前自动被调用
    #     # 此时梯度已经计算好了，非常适合检查梯度问题！
    #
    #     print("--- Gradient Check ---")
    #     # 检查你的 MLP 层的梯度
    #     for name, param in self.transformer.custom_time_text_embed.time_fusion_mlp.named_parameters():
    #         if param.grad is None:
    #             print(f"❌ {name}: Grad is None! (Disconnected graph)")
    #         else:
    #             grad_mean = param.grad.abs().mean().item()
    #             grad_max = param.grad.abs().max().item()
    #             print(f"✅ {name}: Grad Mean={grad_mean:.2e}, Max={grad_max:.2e}")
    #
    #             if grad_mean == 0:
    #                 print(f"⚠️ {name}: Grad is ZERO! (Vanishing gradient or precision issue)")
    #
    #         # 只看前几个就行，不用打印全部
    #         break

    @staticmethod
    def replace_and_freeze_time_text_embed(fluxpipe, time_layers_path):
        """
        替换 transformer.time_text_embed 为 time_text_embed_module2
        仅训练 timestep_embedder 和 delta_timestep_embedder
        - timestep_embedder: 从原模型加载权重
        - delta_timestep_embedder: 零初始化
        args:
            fluxpipe: FLUX 管道对象（包含 transformer 模块）
            time_layers_path: 自定义时间层权重路径（可选，当前逻辑下仅兼容 embedder 权重）
        """
        # 1. 获取原模块（用于继承 timestep_embedder 权重）

        original_embed_module = fluxpipe.transformer.time_text_embed

        # 2. 获取维度参数（与原模块对齐）
        embedding_dim = fluxpipe.transformer.inner_dim
        pooled_projection_dim = fluxpipe.transformer.config.pooled_projection_dim

        # 3. 实例化修改后的自定义模块（time_text_embed_module2）
        custom_embed_module = time_text_embed_module2(
            embedding_dim=embedding_dim,
            pooled_projection_dim=pooled_projection_dim
        )

        # 4. 权重复用与初始化（核心要求）
        # 4.1 加载 time_proj 权重（t 编码的投影层，后续冻结，仅保证结构对齐）
        custom_embed_module.time_proj.load_state_dict(original_embed_module.time_proj.state_dict())
        # 4.2 timestep_embedder 加载原模型权重（符合要求：继承原权重）
        custom_embed_module.timestep_embedder.load_state_dict(original_embed_module.timestep_embedder.state_dict())
        # 4.3 delta_time_proj 复用 time_proj 权重（结构一致，后续冻结，不影响训练）
        custom_embed_module.delta_time_proj.load_state_dict(original_embed_module.time_proj.state_dict())

        # 4.4 delta_timestep_embedder 零初始化（关键：不加载原权重，强制置零）
        def zero_init_module(module):
            """辅助函数：将模块所有可学习参数置零"""
            for param in module.parameters():
                nn.init.constant_(param, 0.0)

        zero_init_module(custom_embed_module.delta_timestep_embedder)
        # 4.5 加载 text_embedder 权重（文本编码层，后续冻结）
        custom_embed_module.text_embedder.load_state_dict(original_embed_module.text_embedder.state_dict())

        # 5. 冻结非目标层，仅解冻两个 embedder（核心训练目标）
        # 5.1 冻结 time_proj（t 投影层，无训练需求）
        for param in custom_embed_module.time_proj.parameters():
            param.requires_grad = False
        # 5.2 冻结 delta_time_proj（Δt 投影层，无训练需求）
        for param in custom_embed_module.delta_time_proj.parameters():
            param.requires_grad = False
        # 5.3 冻结 text_embedder（文本编码层，无训练需求）
        for param in custom_embed_module.text_embedder.parameters():
            param.requires_grad = False

        # 5.4 显式设置两个 embedder 可训练（核心：仅这两个层参与训练）
        # 解冻 timestep_embedder
        for param in custom_embed_module.timestep_embedder.parameters():
            param.requires_grad = True
        # 解冻 delta_timestep_embedder（零初始化后，开启训练）
        for param in custom_embed_module.delta_timestep_embedder.parameters():
            param.requires_grad = True

        # 6. 替换原模块（保持原有挂载方式，兼容后续逻辑）
        fluxpipe.transformer.custom_time_text_embed = custom_embed_module

        # 7. 加载自定义时间层权重（若路径不为 None，仅适配两个 embedder 权重）
        # 注意：若使用之前分开保存的 MLP 权重，此处会报错，需对应更新 OminiModel.load_custom_embed_weights
        if time_layers_path is not None:
            OminiModel.load_custom_embed_weights(fluxpipe.transformer, time_layers_path)

        # 8. 整理并返回所有可训练参数（两个 embedder 的参数列表合并）
        trainable_params = []
        trainable_params.extend(list(custom_embed_module.timestep_embedder.parameters()))
        trainable_params.extend(list(custom_embed_module.delta_timestep_embedder.parameters()))

        return trainable_params
    # def replace_and_freeze_time_text_embed(fluxpipe, time_layers_path):
    #     """
    #     替换 transformer.time_text_embed 为 time_text_embed_module2（兼容模型并行，解决 Meta 张量 no data 问题）
    #     仅训练 timestep_embedder 和 delta_timestep_embedder
    #     - timestep_embedder: 从原模型加载权重（先落地 Meta 层，提取有效权重）
    #     - delta_timestep_embedder: 零初始化
    #     args:
    #         fluxpipe: FLUX 管道对象（包含 transformer 模块，device_map="balanced" 两张卡并行）
    #         time_layers_path: 自定义时间层权重路径（可选，当前逻辑下仅兼容 embedder 权重）
    #     """
    #
    #     # ------------- 新增：核心辅助函数（解决 Meta 张量问题，适配两张卡并行）-------------
    #     def get_valid_device_and_dtype(module):
    #         """从模块中自动提取有效设备（非 meta）和 dtype，避免硬编码 GPU"""
    #         for param in module.parameters():
    #             if not param.is_meta and param.device != torch.device('meta'):
    #                 return param.device, param.dtype
    #         # 兜底：两张卡默认使用 cuda:0，也可改为 cuda:1
    #         return torch.device("cuda:0"), torch.float32
    #
    #     def force_land_meta_module(original_module, flux_pipe, target_device, target_dtype):
    #         """主动将 Meta 设备的原模块落地到具体 GPU，加载真实权重数据"""
    #         if original_module is None:
    #             raise ValueError("需要提取权重的原模块不能为 None")
    #
    #         # 1. 判断是否已落地（非 meta 设备，有有效数据）
    #         has_valid_data = False
    #         for param in original_module.parameters():
    #             if not param.is_meta and param.device != torch.device('meta'):
    #                 has_valid_data = True
    #                 break
    #         if has_valid_data:
    #             print(f"[Meta 层落地] 原模块已在有效设备 {target_device}，无需重复落地")
    #             return original_module.to(target_device, dtype=target_dtype, non_blocking=True)
    #
    #         # 2. 核心：利用 accelerate 加载本地权重并分发到目标 GPU，解决 no data 问题
    #         try:
    #             landed_module = load_checkpoint_and_dispatch(
    #                 original_module,
    #                 checkpoint=LOCAL_FLUX_DIR,  # 对应 LOCAL_FLUX_DIR
    #                 device_map={"": target_device},  # 仅将该模块分发到目标 GPU，不破坏整体并行
    #                 dtype=target_dtype,
    #                 local_files_only=True,
    #                 skip_keys=None
    #             )
    #             print(f"[Meta 层落地] 原模块已成功落地到 {target_device}，加载真实权重")
    #             return landed_module
    #         except Exception as e:
    #             print(f"[Meta 层落地] 自动加载失败，兜底迁移设备：{e}")
    #             # 兜底：手动迁移设备（仅分配内存，无真实数据，避免流程中断）
    #             original_module.to_empty(device=target_device)
    #             return original_module
    #
    #     def safe_extract_state_dict(module):
    #         """从落地后的模块中安全提取权重，过滤残留 Meta 张量"""
    #         valid_state_dict = {}
    #         raw_state_dict = module.state_dict()
    #         for key, param in raw_state_dict.items():
    #             if not param.is_meta and param.device != torch.device('meta'):
    #                 # 克隆参数，避免修改原模块权重
    #                 valid_state_dict[key] = param.detach().clone()
    #             else:
    #                 print(f"[安全提权] 跳过残留 Meta 张量参数：{key}")
    #         return valid_state_dict
    #
    #     # ------------- 步骤 1：提取有效设备和 dtype，适配两张卡并行 -------------
    #     valid_device, valid_dtype = get_valid_device_and_dtype(fluxpipe.transformer)
    #     print(f"[流程初始化] 提取到有效设备：{valid_device}，dtype：{valid_dtype}")
    #
    #     # ------------- 步骤 2：获取原模块并主动落地 Meta 层（核心：解决 no data）-------------
    #     original_embed_module = fluxpipe.transformer.time_text_embed
    #     # 主动落地原模块，加载真实权重数据
    #     landed_original_embed_module = force_land_meta_module(
    #         original_embed_module,
    #         fluxpipe,
    #         valid_device,
    #         valid_dtype
    #     )
    #
    #     # ------------- 步骤 3：获取维度参数（与原模块对齐，保留原有逻辑）-------------
    #     embedding_dim = fluxpipe.transformer.inner_dim
    #     pooled_projection_dim = fluxpipe.transformer.config.pooled_projection_dim
    #
    #     # ------------- 步骤 4：实例化自定义模块（落地到有效 GPU，避免 Meta 占位）-------------
    #     custom_embed_module = time_text_embed_module2(
    #         embedding_dim=embedding_dim,
    #         pooled_projection_dim=pooled_projection_dim
    #     ).to(  # 直接迁移到有效设备，兼容两张卡并行
    #         device=valid_device,
    #         dtype=valid_dtype,
    #         non_blocking=True
    #     )
    #
    #     # ------------- 步骤 5：权重复用与初始化（核心修改：安全提取权重，解决 Meta 问题）-------------
    #     # 5.1 安全提取原模块各组件的有效权重（避免 no data 报错）
    #     original_time_proj_sd = safe_extract_state_dict(landed_original_embed_module.time_proj)
    #     original_timestep_embed_sd = safe_extract_state_dict(landed_original_embed_module.timestep_embedder)
    #     original_text_embed_sd = safe_extract_state_dict(landed_original_embed_module.text_embedder)
    #
    #     # 5.2 加载 time_proj 权重（t 编码的投影层，后续冻结，仅保证结构对齐）
    #     if original_time_proj_sd:
    #         custom_embed_module.time_proj.load_state_dict(original_time_proj_sd, strict=False)
    #     else:
    #         print(f"[权重加载] time_proj 无有效权重，使用默认初始化")
    #
    #     # 5.3 timestep_embedder 加载原模型权重（符合要求：继承原权重，兼容 Meta 张量）
    #     if original_timestep_embed_sd:
    #         custom_embed_module.timestep_embedder.load_state_dict(original_timestep_embed_sd, strict=False)
    #     else:
    #         print(f"[权重加载] timestep_embedder 无有效权重，使用默认初始化")
    #
    #     # 5.4 delta_time_proj 复用 time_proj 权重（结构一致，后续冻结，不影响训练）
    #     if original_time_proj_sd:
    #         custom_embed_module.delta_time_proj.load_state_dict(original_time_proj_sd, strict=False)
    #     else:
    #         print(f"[权重加载] delta_time_proj 无有效权重，使用默认初始化")
    #
    #     # 5.5 delta_timestep_embedder 零初始化（关键：不加载原权重，强制置零，保留原有逻辑）
    #     def zero_init_module(module):
    #         """辅助函数：将模块所有可学习参数置零（兼容 Meta 张量，仅操作有效参数）"""
    #         for param in module.parameters():
    #             if not param.is_meta:  # 跳过 Meta 张量，避免报错
    #                 nn.init.constant_(param, 0.0)
    #
    #     zero_init_module(custom_embed_module.delta_timestep_embedder)
    #
    #     # 5.6 加载 text_embedder 权重（文本编码层，后续冻结，兼容 Meta 张量）
    #     if original_text_embed_sd:
    #         custom_embed_module.text_embedder.load_state_dict(original_text_embed_sd, strict=False)
    #     else:
    #         print(f"[权重加载] text_embedder 无有效权重，使用默认初始化")
    #
    #     # ------------- 步骤 6：冻结非目标层，仅解冻两个 embedder（保留原有逻辑，增加 Meta 兼容）-------------
    #     def safe_freeze_module(module):
    #         """安全冻结模块，跳过 Meta 张量，避免报错"""
    #         for param in module.parameters():
    #             if not param.is_meta:
    #                 param.requires_grad = False
    #
    #     def safe_unfreeze_module(module):
    #         """安全解冻模块，跳过 Meta 张量，避免报错"""
    #         for param in module.parameters():
    #             if not param.is_meta:
    #                 param.requires_grad = True
    #
    #     # 6.1 冻结 time_proj（t 投影层，无训练需求）
    #     safe_freeze_module(custom_embed_module.time_proj)
    #     # 6.2 冻结 delta_time_proj（Δt 投影层，无训练需求）
    #     safe_freeze_module(custom_embed_module.delta_time_proj)
    #     # 6.3 冻结 text_embedder（文本编码层，无训练需求）
    #     safe_freeze_module(custom_embed_module.text_embedder)
    #
    #     # 6.4 显式设置两个 embedder 可训练（核心：仅这两个层参与训练）
    #     # 解冻 timestep_embedder
    #     safe_unfreeze_module(custom_embed_module.timestep_embedder)
    #     # 解冻 delta_timestep_embedder（零初始化后，开启训练）
    #     safe_unfreeze_module(custom_embed_module.delta_timestep_embedder)
    #
    #     # ------------- 步骤 7：替换原模块（保持原有挂载方式，兼容模型并行）-------------
    #     # 关键：绑定前再次确认设备，保证自定义模块适配两张卡的并行分布
    #     fluxpipe.transformer.custom_time_text_embed = custom_embed_module.to(
    #         device=valid_device,
    #         dtype=valid_dtype,
    #         non_blocking=True
    #     )
    #     print(f"[模块替换] 自定义模块 custom_time_text_embed 已成功绑定到 transformer")
    #
    #     # ------------- 步骤 8：加载自定义时间层权重（保留原有逻辑，增加 Meta 兼容）-------------
    #     # 注意：若使用之前分开保存的 MLP 权重，此处会报错，需对应更新 OminiModel.load_custom_embed_weights
    #     if time_layers_path is not None and isinstance(time_layers_path, str):
    #         # 传递有效设备，避免加载权重时出现 Meta 张量/设备不匹配问题
    #         OminiModel.load_custom_embed_weights(
    #             fluxpipe.transformer,
    #             time_layers_path,
    #             map_location=valid_device  # 新增：适配模型并行，需保证 load_custom_embed_weights 支持该参数
    #         )
    #
    #     # ------------- 步骤 9：整理并返回所有可训练参数（过滤 Meta 张量，兼容并行）-------------
    #     trainable_params = []
    #
    #     def collect_valid_trainable_params(module, param_list):
    #         """收集有效可训练参数，跳过 Meta 张量，避免无效参数传入优化器"""
    #         for param in module.parameters():
    #             if param.requires_grad and not param.is_meta:
    #                 param_list.append(param)
    #
    #     collect_valid_trainable_params(custom_embed_module.timestep_embedder, trainable_params)
    #     collect_valid_trainable_params(custom_embed_module.delta_timestep_embedder, trainable_params)
    #
    #     print(f"[流程完成] 可训练参数总数：{len(trainable_params)}，已成功返回")
    #     return trainable_params



    def verify_module_params(self):
        custom_module=self.transformer.custom_time_text_embed
        """验证自定义模块的参数可训练状态"""
        print("=== 自定义模块参数可训练状态 ===")
        for name, param in custom_module.named_parameters():
            print(f"参数名: {name:<60} 可训练: {param.requires_grad}")

        # 统计可训练参数
        trainable_params = sum(p.numel() for p in custom_module.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in custom_module.parameters())
        print(f"\n可训练参数总数: {trainable_params:,} / 总参数数: {total_params:,}")

    def _print_trainable_params_details(self):
        """
        辅助函数：打印所有可训练参数的详细信息，包括：
        - 参数名称
        - 参数形状
        - 参数类型（LoRA/自定义 time_fusion_mlp/其他）
        - 可训练参数总数/总参数量
        """
        # 收集可训练参数的名称、形状、类型
        trainable_params_info = []
        total_trainable_params = 0  # 可训练参数总数（元素个数）
        total_model_params = 0  # 模型总参数数（元素个数）

        for name, param in self.transformer.named_parameters():
            # 统计模型总参数数
            total_model_params += param.numel()

            # 仅处理可训练参数
            if param.requires_grad:
                # 判断参数类型
                if "time_fusion_mlp" in name:
                    param_type = "自定义 time_fusion_mlp"
                    # 记录参数信息
                    trainable_params_info.append({
                        "name": name,
                        "shape": list(param.shape),
                        "type": param_type,
                        "numel": param.numel()  # 该参数的元素个数
                    })
                elif "custom_time_text_embed" in name:
                    param_type = "自定义 time_text_embed（非融合层）"
                    # 记录参数信息
                    trainable_params_info.append({
                        "name": name,
                        "shape": list(param.shape),
                        "type": param_type,
                        "numel": param.numel()  # 该参数的元素个数
                    })



                # 累加可训练参数总数
                total_trainable_params += param.numel()

        # 打印标题
        print("\n" + "=" * 80)
        print("📌 可训练参数详情")
        print("=" * 80)

        # 打印每个可训练参数
        if trainable_params_info:
            for idx, info in enumerate(trainable_params_info, 1):
                print(f"\n[{idx}] 参数名称: {info['name']}")
                print(f"   形状: {info['shape']}")
                print(f"   类型: {info['type']}")
                print(f"   元素个数: {info['numel']:,}")
        else:
            print("\n⚠️  未检测到任何可训练参数！")

        # 打印统计信息
        print("\n" + "-" * 80)
        print(f"📊 统计汇总:")
        print(f"   可训练参数数量（个）: {len(trainable_params_info)}")
        print(f"   可训练参数总元素数: {total_trainable_params:,}")
        print(f"   模型总参数元素数: {total_model_params:,}")
        print(f"   可训练参数占比: {total_trainable_params / total_model_params * 100:.4f}%")
        print("=" * 80 + "\n")



    def init_lora(self, lora_path: str, lora_config: dict, device):
        assert lora_path or lora_config
        if lora_path:
            for adapter_name in self.adapter_set:
                lora_file = os.path.join(lora_path, f"{adapter_name}.safetensors")

                if not os.path.exists(lora_file):
                    raise FileNotFoundError(f"LoRA file not found: {lora_file}")

                # 使用传入的 lora_config 添加适配器
                self.transformer.add_adapter(
                    LoraConfig(**lora_config), adapter_name=adapter_name
                )

                # 加载权重
                lora_state_dict = load_file(lora_file)
                set_peft_model_state_dict(
                    self.transformer,
                    lora_state_dict,
                    adapter_name=adapter_name
                )

            lora_layers = filter(
                lambda p: p.requires_grad, self.transformer.parameters()
            )
            print(f"load lora from {lora_path}")
        else:
            for adapter_name in self.adapter_set:
                self.transformer.add_adapter(
                    LoraConfig(**lora_config), adapter_name=adapter_name
                )

            # TODO: Check if this is correct (p.requires_grad)
            lora_layers = filter(
                lambda p: p.requires_grad, self.transformer.parameters()
            )
            # if device=="cuda:0":
            #     self._print_trainable_params_details()
        return list(lora_layers)

    def save_lora(self, path: str):
        for adapter_name in self.adapter_set:
            FluxPipeline.save_lora_weights(
                save_directory=path,
                weight_name=f"{adapter_name}.safetensors",
                transformer_lora_layers=get_peft_model_state_dict(
                    self.transformer, adapter_name=adapter_name
                ),
                safe_serialization=True,
            )

    def save_custom_embed_weights(self, save_directory: str,
                                  timestep_embedder_weight_name: str = "timestep_embedder_weights.safetensors",
                                  delta_timestep_embedder_weight_name: str = "delta_timestep_embedder_weights.safetensors"):
        """
        分开保存自定义模块中两个可训练 embedder 的权重（各自为独立文件）
        适配 time_text_embed_module2，保存 timestep_embedder 和 delta_timestep_embedder
        args:
            save_directory: 保存目录
            timestep_embedder_weight_name: timestep_embedder 的权重文件名
            delta_timestep_embedder_weight_name: delta_timestep_embedder 的权重文件名
        """
        # 1. 检查自定义模块是否存在
        if not hasattr(self.transformer, "custom_time_text_embed"):
            print("Warning: custom_time_text_embed not found, skipping save.")
            return

        # 2. 获取自定义模块
        custom_module = self.transformer.custom_time_text_embed

        # 3. 确保保存目录存在
        os.makedirs(save_directory, exist_ok=True)

        # 4. 保存 timestep_embedder 权重（独立文件，替换原 MLP 逻辑）
        timestep_embedder_state = custom_module.timestep_embedder.state_dict()
        timestep_save_path = os.path.join(save_directory, timestep_embedder_weight_name)
        save_file(timestep_embedder_state, timestep_save_path)
        print(f"Timestep Embedder weights saved to {timestep_save_path}")

        # 5. 保存 delta_timestep_embedder 权重（独立文件，替换原 MLP 逻辑）
        delta_timestep_embedder_state = custom_module.delta_timestep_embedder.state_dict()
        delta_timestep_save_path = os.path.join(save_directory, delta_timestep_embedder_weight_name)
        save_file(delta_timestep_embedder_state, delta_timestep_save_path)
        print(f"Delta Timestep Embedder weights saved to {delta_timestep_save_path}")

    @staticmethod
    def load_custom_embed_weights(transformer, load_directory: str,
                                  timestep_embedder_weight_name: str = "timestep_embedder_weights.safetensors",
                                  delta_timestep_embedder_weight_name: str = "delta_timestep_embedder_weights.safetensors"):
        """
        分开加载两个独立 embedder 的权重（对应两个独立文件，适配 time_text_embed_module2）
        注意：必须先调用 replace_and_freeze_time_text_embed 初始化结构后，才能调用此方法加载权重
        """
        # 1. 检查结构是否已经初始化
        if not hasattr(transformer, "custom_time_text_embed"):
            raise RuntimeError(
                "custom_time_text_embed not initialized. "
                "Please run `replace_and_freeze_time_text_embed` before loading weights."
            )

        # 2. 获取自定义模块
        custom_module = transformer.custom_time_text_embed

        # 3. 加载 timestep_embedder 权重（独立文件，替换原 MLP 逻辑）
        # 3.1 检查 timestep Embedder 权重文件是否存在
        timestep_load_path = os.path.join(load_directory, timestep_embedder_weight_name)
        if not os.path.exists(timestep_load_path):
            raise FileNotFoundError(f"Timestep Embedder weight file not found: {timestep_load_path}")
        # 3.2 加载并写入 timestep_embedder
        timestep_embedder_state = load_file(timestep_load_path)
        msg1 = custom_module.timestep_embedder.load_state_dict(timestep_embedder_state, strict=True)

        # 4. 加载 delta_timestep_embedder 权重（独立文件，替换原 MLP 逻辑）
        # 4.1 检查 delta_timestep Embedder 权重文件是否存在
        delta_timestep_load_path = os.path.join(load_directory, delta_timestep_embedder_weight_name)
        if not os.path.exists(delta_timestep_load_path):
            raise FileNotFoundError(f"Delta Timestep Embedder weight file not found: {delta_timestep_load_path}")
        # 4.2 加载并写入 delta_timestep_embedder
        delta_timestep_embedder_state = load_file(delta_timestep_load_path)
        msg2 = custom_module.delta_timestep_embedder.load_state_dict(delta_timestep_embedder_state, strict=True)

        # 5. 打印加载结果
        print(f"Loaded all custom time Embedder weights from {load_directory}")
        print(f"Timestep Embedder load result: {msg1}")
        print(f"Delta Timestep Embedder load result: {msg2}")

    def configure_optimizers(self):
        # Freeze the transformer
        self.transformer.requires_grad_(False)
        opt_config = self.optimizer_config

        # Set the trainable parameters
        self.trainable_params = self.lora_layers + self.time_layers
        # self.trainable_params = self.lora_layers
        # Unfreeze trainable parameters
        for p in self.trainable_params:
            p.requires_grad_(True)

        # mlp_lr = opt_config["params"]["lr"] * 2
        # param_groups = [
        #     {
        #         "params": self.time_layers,
        #         "lr": mlp_lr,
        #         "betas": opt_config["params"]["betas"],
        #         "weight_decay": opt_config["params"]["weight_decay"]
        #     },
        #     {
        #         "params": self.lora_layers,
        #         **opt_config["params"],# 沿用原有配置
        #     }
        # ]
        # # Initialize the optimizer
        if opt_config["type"] == "AdamW":
            optimizer = torch.optim.AdamW(self.trainable_params, **opt_config["params"])
            # optimizer = torch.optim.AdamW(param_groups)
        elif opt_config["type"] == "Prodigy":
            optimizer = prodigyopt.Prodigy(
                self.trainable_params,
                **opt_config["params"],
            )
        elif opt_config["type"] == "SGD":
            optimizer = torch.optim.SGD(self.trainable_params, **opt_config["params"])
        else:
            raise NotImplementedError("Optimizer not implemented.")
        return optimizer

    # def training_step(self, batch, batch_idx):
    #     imgs, prompts = batch["image"], batch["description"]
    #     image_latent_mask = batch.get("image_latent_mask", None)
    #
    #     # Get the conditions and position deltas from the batch
    #     conditions, position_deltas, position_scales, latent_masks = [], [], [], []
    #     for i in range(1000):
    #         if f"condition_{i}" not in batch:
    #             break
    #         conditions.append(batch[f"condition_{i}"])
    #         position_deltas.append(batch.get(f"position_delta_{i}", [[0, 0]]))
    #         position_scales.append(batch.get(f"position_scale_{i}", [1.0])[0])
    #         latent_masks.append(batch.get(f"condition_latent_mask_{i}", None))
    #
    #     # Prepare inputs
    #     with torch.no_grad():
    #         # Prepare image input
    #         x_0, img_ids = encode_images(self.flux_pipe, imgs)
    #
    #         # Prepare text input
    #         (
    #             prompt_embeds,
    #             pooled_prompt_embeds,
    #             text_ids,
    #         ) = self.flux_pipe.encode_prompt(
    #             prompt=prompts,
    #             prompt_2=None,
    #             prompt_embeds=None,
    #             pooled_prompt_embeds=None,
    #             device=self.flux_pipe.device,
    #             num_images_per_prompt=1,
    #             max_sequence_length=self.model_config.get("max_sequence_length", 512),
    #             lora_scale=None,
    #         )
    #
    #         # Prepare t and x_t
    #         t = torch.sigmoid(torch.randn((imgs.shape[0],), device=self.device))
    #         x_1 = torch.randn_like(x_0).to(self.device)
    #         t_ = t.unsqueeze(1).unsqueeze(1)
    #         x_t = ((1 - t_) * x_0 + t_ * x_1).to(self.dtype)
    #         if image_latent_mask is not None:
    #             x_0 = x_0[:, image_latent_mask[0]]
    #             x_1 = x_1[:, image_latent_mask[0]]
    #             x_t = x_t[:, image_latent_mask[0]]
    #             img_ids = img_ids[image_latent_mask[0]]
    #
    #         # Prepare conditions
    #         condition_latents, condition_ids = [], []
    #         for cond, p_delta, p_scale, latent_mask in zip(
    #             conditions, position_deltas, position_scales, latent_masks
    #         ):
    #             # Prepare conditions
    #             c_latents, c_ids = encode_images(self.flux_pipe, cond)
    #             # Scale the position (see OminiConrtol2)
    #             if p_scale != 1.0:
    #                 scale_bias = (p_scale - 1.0) / 2
    #                 c_ids[:, 1:] *= p_scale
    #                 c_ids[:, 1:] += scale_bias
    #             # Add position delta (see OminiControl)
    #             c_ids[:, 1] += p_delta[0][0]
    #             c_ids[:, 2] += p_delta[0][1]
    #             if len(p_delta) > 1:
    #                 print("Warning: only the first position delta is used.")
    #             # Append to the list
    #             if latent_mask is not None:
    #                 c_latents, c_ids = c_latents[latent_mask], c_ids[latent_mask[0]]
    #             condition_latents.append(c_latents)
    #             condition_ids.append(c_ids)
    #
    #         # Prepare guidance
    #         guidance = (
    #             torch.ones_like(t).to(self.device)
    #             if self.transformer.config.guidance_embeds #默认false
    #             else None
    #         )
    #
    #     branch_n = 2 + len(conditions)
    #     group_mask = torch.ones([branch_n, branch_n], dtype=torch.bool).to(self.device)
    #     # Disable the attention cross different condition branches
    #     group_mask[2:, 2:] = torch.diag(torch.tensor([1] * len(conditions)))
    #     # Disable the attention from condition branches to image branch and text branch
    #     if self.model_config.get("independent_condition", False):
    #         group_mask[2:, :2] = False
    #
    #     # Forward pass
    #     transformer_out = transformer_forward(
    #         self.transformer,
    #         image_features=[x_t, *(condition_latents)],
    #         text_features=[prompt_embeds],
    #         img_ids=[img_ids, *(condition_ids)],
    #         txt_ids=[text_ids],
    #         # There are three timesteps for the three branches
    #         # (text, image, and the condition)
    #         timesteps=[t, t] + [torch.zeros_like(t)] * len(conditions),
    #         # Same as above
    #         pooled_projections=[pooled_prompt_embeds] * branch_n,
    #         guidances=[guidance] * branch_n,
    #         # The LoRA adapter names of each branch
    #         adapters=self.adapter_names,
    #         return_dict=False,
    #         group_mask=group_mask,
    #     )
    #     pred = transformer_out[0]
    #
    #     # Compute loss
    #     step_loss = torch.nn.functional.mse_loss(pred, (x_1 - x_0), reduction="mean")
    #     self.last_t = t.mean().item()
    #
    #     self.log_loss = (
    #         step_loss.item()
    #         if not hasattr(self, "log_loss")
    #         else self.log_loss * 0.95 + step_loss.item() * 0.05
    #     )
    #     return step_loss
    def training_step(self, batch, batch_idx):
        imgs, prompts = batch["image"], batch["description"]
        image_latent_mask = batch.get("image_latent_mask", None)

        # Get the conditions and position deltas from the batch
        conditions, position_deltas, position_scales, latent_masks = [], [], [], []
        for i in range(1000):
            if f"condition_{i}" not in batch:
                break
            conditions.append(batch[f"condition_{i}"])
            position_deltas.append(batch.get(f"position_delta_{i}", [[0, 0]]))
            position_scales.append(batch.get(f"position_scale_{i}", [1.0])[0])
            latent_masks.append(batch.get(f"condition_latent_mask_{i}", None))
        local_rank = get_rank()
        # Prepare inputs
        with torch.no_grad():
            # Prepare image input
            x_0, img_ids = encode_images(self.flux_pipe, imgs)
            x_0 = x_0.to(self.device)
            img_ids = img_ids.to(self.device)
            # Prepare text input
            (
                prompt_embeds,
                pooled_prompt_embeds,
                text_ids,
            ) = self.flux_pipe.encode_prompt(
                prompt=prompts,
                prompt_2=None,
                prompt_embeds=None,
                pooled_prompt_embeds=None,
                device=self.flux_pipe.device,
                num_images_per_prompt=1,
                max_sequence_length=self.model_config.get("max_sequence_length", 512),
                lora_scale=None,
            )

            # -------------------------- Mean Flows 改动1: t, r 采样（lognorm(-0.4, 1.0) + 25% r≠t）--------------------------
            # 1. 定义 logit-normal 采样器（先采样正态分布，再通过logistic函数映射到(0,1)）
            def lognorm_sampler(batch_size, mu=-0.4, sigma=1.0, device=None):
                normal_dist = Normal(mu, sigma)
                logits = normal_dist.sample((batch_size,))
                return torch.sigmoid(logits).to(device)

            batch_size = imgs.shape[0]
            # 采样 t 和 r（独立采样后保证 t > r）
            t_raw = lognorm_sampler(batch_size, device=self.device)
            r_raw = lognorm_sampler(batch_size, device=self.device)
            t = torch.max(t_raw, r_raw)
            r = torch.min(t_raw, r_raw)
            # print(f"before:t:{t},{t.dtype},{t.shape}")
            # print(f"before:r:{r},{r.dtype},{r.shape}")
            # 25% 概率让 r ≠ t（论文Tab.1a最优配置）
            r_eq_t_mask = torch.rand(batch_size, device=self.device) > 0.25
            r[r_eq_t_mask] = t[r_eq_t_mask]
            r = r.to(self.flux_pipe.dtype)
            t = t.to(self.flux_pipe.dtype)
            # print(f"after:t:{t},{t.dtype},{t.shape},{self.flux_pipe.dtype}")
            # print(f"after:r:{r},{r.dtype},{r.shape}")
            # -------------------------- 原逻辑保留：x_t 计算 --------------------------
            x_1 = torch.randn_like(x_0).to(self.device)
            t_ = t.unsqueeze(1).unsqueeze(1) # 适配 latent 维度 (B, 1, 1)
            # print(f"x_0 设备：{x_0.device}")
            # print(f"x_1 设备：{x_1.device}")
            # print(f"t_ 设备：{t_.device}")
            x_t = ((1 - t_) * x_0 + t_ * x_1).to(self.flux_pipe.dtype)
            #print(f"x_t:{x_t.dtype}")
            # r=t
            if image_latent_mask is not None:
                x_0 = x_0[:, image_latent_mask[0]]
                x_1 = x_1[:, image_latent_mask[0]]
                x_t = x_t[:, image_latent_mask[0]]
                img_ids = img_ids[image_latent_mask[0]]

            # Prepare conditions
            img_size = (imgs.shape[2], imgs.shape[3])
            #print(f"empty_image:{img_size}")
            condition_empty = Image.new("RGB", img_size, (0, 0, 0))
            condition_latents, uc_latents,condition_ids = [], [], []
            for cond, p_delta, p_scale, latent_mask in zip(
                conditions, position_deltas, position_scales, latent_masks
            ):
                # Prepare conditions
                c_latents, c_ids = encode_images(self.flux_pipe, cond)
                c_latents = c_latents.to(self.device)
                c_ids = c_ids.to(self.device)
                # Scale the position (see OminiConrtol2)
                if p_scale != 1.0:
                    scale_bias = (p_scale - 1.0) / 2
                    c_ids[:, 1:] *= p_scale
                    c_ids[:, 1:] += scale_bias
                # Add position delta (see OminiControl)
                # c_ids[:, 1] += p_delta[0][0]
                # c_ids[:, 2] += p_delta[0][1]
                # if len(p_delta) > 1:
                #     print("Warning: only the first position delta is used.")
                # Append to the list
                if latent_mask is not None:
                    c_latents, c_ids = c_latents[latent_mask], c_ids[latent_mask[0]]
                condition_latents.append(c_latents)
                condition_ids.append(c_ids)
                uc_latents.append(encode_images(self.flux_pipe, condition_empty)[0].expand(batch_size, -1, -1))

            # Prepare guidance
            guidance = (
                torch.ones_like(t).to(self.device)
                if self.transformer.config.guidance_embeds
                else None
            )

        branch_n = 2 + len(conditions)
        group_mask = torch.ones([branch_n, branch_n], dtype=torch.bool).to(self.device)
        # Disable the attention cross different condition branches
        group_mask[2:, 2:] = torch.diag(torch.tensor([1] * len(conditions)))
        # Disable the attention from condition branches to image branch and text branch
        if self.model_config.get("independent_condition", False):
            group_mask[2:, :2] = False

        # -------------------------- Mean Flows 改动2: Positional Embedding（t, t-r）--------------------------
        # 计算时间间隔 delta_t = t - r（论文Tab.1c最优配置）
        delta_t = t - r
        v_t = x_1 - x_0
        # print(f"[{local_rank}] Base v_t mean: {v_t.abs().mean().item():.4f}, {v_t.shape}")
        # print(f"delta_t:{delta_t},{delta_t.dtype},{delta_t.shape}")

        # # -------------------------- Forward Pass（适配 Mean Flows 平均速度预测）--------------------------
        # # 模型输出 u_theta：预测平均速度 u(z_t, r, t)
        # transformer_out = transformer_forward(
        #     self.transformer,
        #     image_features=[x_t, *(condition_latents)],
        #     text_features=[prompt_embeds],
        #     img_ids=[img_ids, *(condition_ids)],
        #     txt_ids=[text_ids],
        #     # 传入原始 t（用于模型内部计算），并添加 delta_t 作为位置编码
        #     timesteps=[t, t] + [torch.zeros_like(t)] * len(conditions),
        #     delta_t=[delta_t, delta_t] + [torch.zeros_like(delta_t)] * len(conditions),  # 所有分支共享 delta_t
        #     pooled_projections=[pooled_prompt_embeds] * branch_n,
        #     guidances=[guidance] * branch_n,
        #     adapters=self.adapter_names,
        #     return_dict=False,
        #     group_mask=group_mask,
        # )
        # u_theta = transformer_out[0]  # 模型输出：平均速度预测值

        # def manual_chunked_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None,
        #                              chunk_size=512):
        #     """
        #     手动实现的分块注意力机制 (Memory Efficient Math Attention)。
        #     支持 JVP，且通过分块计算避免 OOM。
        #
        #     Args:
        #         chunk_size: 每次处理的 Query 长度。越小越省显存，但速度稍慢。建议 256-1024。
        #     """
        #     B, H, L, D = query.shape
        #     _, _, S, _ = key.shape
        #
        #     if scale is None:
        #         scale = 1 / math.sqrt(D)
        #
        #     # 1. 准备 Output 容器
        #     output = torch.empty_like(query)
        #
        #     # 2. 只有在 mask 存在时才处理 mask
        #     # attn_mask shape 通常是 (B, 1, L, S) 或 (B, H, L, S)
        #
        #     # 3. 分块循环 (Slicing)
        #     for i in range(0, L, chunk_size):
        #         end = min(i + chunk_size, L)
        #
        #         # [Batch, Heads, Chunk, Dim]
        #         q_chunk = query[:, :, i:end, :]
        #
        #         # (Q @ K.T) * scale -> [Batch, Heads, Chunk, S]
        #         # 使用 torch.matmul 保证 JVP 兼容性
        #         attn_scores = torch.matmul(q_chunk, key.transpose(-1, -2)) * scale
        #
        #         # 处理 Mask
        #         if attn_mask is not None:
        #             # 切片 Mask: mask[:, :, i:end, :]
        #             mask_chunk = attn_mask[:, :, i:end, :]
        #             attn_scores = attn_scores + mask_chunk
        #
        #         if is_causal:
        #             # 如果是 Causal Mask，需要动态生成
        #             # 这里简化处理：通常 Flux 不用 is_causal=True，而是传入 attn_mask
        #             # 如果确实遇到 is_causal=True，建议使用 torch.ones 构造下三角掩码并切片
        #             pass
        #
        #             # Softmax (在最后一个维度 S 上归一化)
        #         attn_probs = F.softmax(attn_scores, dim=-1, dtype=torch.float32)
        #
        #         # Dropout (训练时通常为 0，JVP 也不建议开 Dropout)
        #         if dropout_p > 0.0:
        #             attn_probs = F.dropout(attn_probs, p=dropout_p, training=True)
        #
        #         # (A @ V) -> [Batch, Heads, Chunk, Dim]
        #         o_chunk = torch.matmul(attn_probs, value)
        #
        #         # 写入结果
        #         output[:, :, i:end, :] = o_chunk
        #
        #         # 显式释放临时显存 (虽然 Python 会自动回收，但在高压下这很有用)
        #         del q_chunk, attn_scores, attn_probs, o_chunk
        #
        #     return output
        # @contextmanager
        # def temporary_fp32_execution():
        #     """
        #     终极 Monkey Patch：
        #     1. 劫持 Linear/LayerNorm/GroupNorm/Embedding -> 解决 BF16 vs FP32 类型不匹配。
        #     2. 全局劫持 scaled_dot_product_attention -> 解决 JVP 不支持 FlashAttention 的问题。
        #     """
        #     # ================= 1. 保存原始方法 =================
        #     orig_linear_forward = nn.Linear.forward
        #     orig_layer_norm_forward = nn.LayerNorm.forward
        #     orig_group_norm_forward = nn.GroupNorm.forward
        #     orig_embedding_forward = nn.Embedding.forward
        #
        #     # 关键：保存原始的 SDPA 函数指针
        #     orig_sdpa = F.scaled_dot_product_attention
        #
        #     # ================= 2. 定义 Patch 方法 =================
        #
        #     # [关键] 劫持 SDPA：
        #     # 无论在 checkpoint 内部还是外部，强制包裹在 MATH kernel 上下文中
        #     def new_sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, **kwargs):
        #         # 强制使用分块注意力，chunk_size 可调（如果还OOM，调小这个值，如 256）
        #         return manual_chunked_attention(
        #             query, key, value,
        #             attn_mask=attn_mask,
        #             dropout_p=dropout_p,
        #             is_causal=is_causal,
        #             scale=scale,
        #             chunk_size=128  # <--- 关键调优参数
        #         )
        #
        #     # Linear Patch: 动态转权重
        #     def new_linear_forward(self, input):
        #         if input.dtype == torch.float32 and self.weight is not None and self.weight.dtype != torch.float32:
        #             weight = self.weight.float()
        #             bias = self.bias.float() if self.bias is not None else None
        #             return F.linear(input, weight, bias)
        #         return orig_linear_forward(self, input)
        #
        #     # LayerNorm Patch
        #     def new_layer_norm_forward(self, input):
        #         if input.dtype == torch.float32 and self.weight is not None and self.weight.dtype != torch.float32:
        #             weight = self.weight.float()
        #             bias = self.bias.float() if self.bias is not None else None
        #             return F.layer_norm(input, self.normalized_shape, weight, bias, self.eps)
        #         return orig_layer_norm_forward(self, input)
        #
        #     # GroupNorm Patch
        #     def new_group_norm_forward(self, input):
        #         if input.dtype == torch.float32 and self.weight is not None and self.weight.dtype != torch.float32:
        #             weight = self.weight.float()
        #             bias = self.bias.float() if self.bias is not None else None
        #             return F.group_norm(input, self.num_groups, weight, bias, self.eps)
        #         return orig_group_norm_forward(self, input)
        #
        #     # Embedding Patch
        #     def new_embedding_forward(self, input):
        #         if self.weight is not None and self.weight.dtype != torch.float32:
        #             return F.embedding(
        #                 input, self.weight.float(), self.padding_idx, self.max_norm,
        #                 self.norm_type, self.scale_grad_by_freq, self.sparse
        #             )
        #         return orig_embedding_forward(self, input)
        #
        #     # ================= 3. 应用全局 Patch =================
        #     # 修改类方法
        #     nn.Linear.forward = new_linear_forward
        #     nn.LayerNorm.forward = new_layer_norm_forward
        #     nn.GroupNorm.forward = new_group_norm_forward
        #     nn.Embedding.forward = new_embedding_forward
        #
        #     # 修改函数模块 (这是最重要的一步，覆盖全局命名空间)
        #     F.scaled_dot_product_attention = new_sdpa
        #
        #     try:
        #         yield
        #     finally:
        #         # ================= 4. 还原原始方法 =================
        #         nn.Linear.forward = orig_linear_forward
        #         nn.LayerNorm.forward = orig_layer_norm_forward
        #         nn.GroupNorm.forward = orig_group_norm_forward
        #         nn.Embedding.forward = orig_embedding_forward
        #         F.scaled_dot_product_attention = orig_sdpa
        # # -------------------------- Mean Flows 改动3: 损失函数（MeanFlow Identity）--------------------------
        # def compute_jvp_result(self, x_t, r, t):
        #     """
        #     计算公式: [u(xt,t,t), 0, 1] . [du/dx, du/dr, du/dt]
        #     全程使用 Float32 以保证数值稳定性
        #     """
        #
        #     # ==========================================
        #     # 步骤 1: 准备 Float32 环境
        #     # ==========================================
        #
        #     # # 提取 FP32 权重 (不影响原模型，占用额外显存)
        #     # # 注意：如果显存非常紧张，可以在这里把 tensor 放在 CPU，functional_call 会自动处理(可能会慢)，
        #     # # 或者只转换必要的层。这里假设显存足够。
        #     # params_f32 = {k: v.float() for k, v in self.transformer.named_parameters()}
        #     # buffers_f32 = {k: v.float() for k, v in self.transformer.named_buffers()}
        #
        #     # 准备输入数据为 FP32
        #     xt_f32 = x_t.float()
        #     r_f32 = r.float()
        #     t_f32 = t.float()
        #
        #     # 还需要确保辅助变量 (condition_latents 等) 也是 FP32
        #     # 这里假设你可以访问这些变量，你需要根据实际情况将它们转为 float
        #     # cond_latents_f32 = [c.float() for c in condition_latents]
        #     # prompt_embeds_f32 = prompt_embeds.float()
        #     # ... 其他所有传入 transformer_forward 的 Tensor 都需要是 float32
        #
        #     # ==========================================
        #     # 步骤 2: 定义纯函数 (Pure Function)
        #     # ==========================================
        #
        #     # 定义一个代理类，用于欺骗 transformer_forward
        #     # 当 transformer_forward 调用 model(...) 时，实际上是在执行 functional_call
        #     # class StatelessModel:
        #     #     def __call__(self_, *args, **kwargs):
        #     #         # 关键点：这里强行使用 params_f32 进行前向传播
        #     #         return functional_call(self.transformer, (params_f32, buffers_f32), args, kwargs)
        #     #
        #     #     # 如果 transformer_forward 访问了 config 等属性，代理给原模型
        #     #     def __getattr__(self_, name):
        #     #         return getattr(self.transformer, name)
        #     #
        #     # stateless_model = StatelessModel()
        #     #
        #     # 重写一份逻辑，去掉 .to(bfloat16)，并使用 stateless_model
        def u_theta_pure_f32(z_in, r_in, t_in,use_kernel=True):
            # 这里的输入已经是 float32 了，千万不要再 cast 成 bf16
            delta_t = t_in - r_in

            # 逻辑复用 (假设 cond=True)
             # 确保这些也是 float32
            # print(f"z_in: {z_in.dtype}, r_in: {r_in.dtype}, t_in: {t_in.dtype}")
            # print(f"condition_latents: {condition_latents[0].dtype}, prompt_embeds: {prompt_embeds.dtype}, pooled_projections: {pooled_prompt_embeds.dtype}")
            # 调用 transformer_forward，但在第一个参数传入我们的代理模型
            with torch.autocast("cuda", enabled=True):
                out = transformer_forward(
                    self.transformer,  # <--- 注入点：使用携带 FP32 权重的代理
                    image_features=[z_in, condition_latents[0]],  # z_in 是 JVP 的变量
                    text_features=[prompt_embeds],  # 确保是 float32
                    img_ids=[img_ids, *condition_ids],
                    txt_ids=[text_ids],
                    timesteps=[t_in, t_in] + [torch.zeros_like(t_in)] * len(conditions),
                    delta_t=[delta_t, delta_t] + [torch.zeros_like(delta_t)] * len(conditions),
                    pooled_projections=[pooled_prompt_embeds] * branch_n,  # 确保是 float32
                    guidances=[guidance] * branch_n,
                    adapters=self.adapter_names,
                    return_dict=False,
                    use_kernel=use_kernel,
                    group_mask=group_mask,
                )[0]

            # 确保输出是 float32 (虽然 functional_call 用 float32 权重跑出来通常就是 float32)
            return out
        #
        #
        #
        #     # ==========================================
        #     # 步骤 3: 计算 Tangent Vector (v_x, 0, 1)
        #     # ==========================================
        #
        #     # 计算向量的第一项 u(x_t, t, t)。
        #     # 使用刚定义的纯函数计算，确保精度一致。
        #     with temporary_fp32_execution():
        #         v_x=v_t.float()
        #
        #         v_r = torch.zeros_like(r_f32)
        #         v_t_ = torch.ones_like(t_f32)
        #
        #         # ==========================================
        #         # 步骤 4: 执行 JVP
        #         # ==========================================
        #
        #         primals = (xt_f32, r_f32, t_f32)
        #         tangents = (v_x, v_r, v_t_)
        #         if hasattr(self.transformer, 'gradient_checkpointing') and not self.transformer.gradient_checkpointing:
        #             self.transformer.enable_gradient_checkpointing()
        #         # u_val 是函数值，jvp_val 是你需要的结果
        #         u_val, jvp_val = jvp(u_theta_pure_f32, primals, tangents)
        #
        #     # 如果后续流程需要 bf16，可以在这里转回，否则返回 float32
        #     return u_val.bfloat16(),jvp_val.bfloat16()
        #
        #
        # u_val, dudt_ = compute_jvp_result(self, x_t, r, t)
        # print(f"[{local_rank}] Base u_val mean: {u_val.abs().mean().item():.4f}, {u_val.shape}")
        # print(f"[{local_rank}] Base dudt_ mean: {dudt_.abs().mean().item():.4f}, {dudt_.shape}")
        # 2. 计算 dudt = 总导数（使用 JVP 高效计算，论文4.1公式8）
        # 定义辅助函数：输入 (z, t, delta_t)，输出 u_theta
        def u_theta_cfg_fn(z, r_in, t_in, cond=True, use_kernel=False):
            """
            计算带CFG的u_theta：区分条件/无条件输出
            :param z: 含噪样本 z_t
            :param r_in: 起始时间 r
            :param t_in: 当前时间 t
            :param cond: True=类别条件输出，False=类别无条件输出
            :return: u_theta^{cfg}(z_t, t, t | c) 或 u_theta^{cfg}(z_t, t, t)
            """
            # z = z.to(torch.bfloat16)
            # r_in = r_in.to(torch.bfloat16)
            # t_in = t_in.to(torch.bfloat16)
            delta_t = t_in - r_in
            # 条件开关：cond=True时传入类别条件，False时清空
            _condition_latents = condition_latents if cond else uc_latents

            with torch.autocast("cuda", enabled=True):
                out = transformer_forward(
                    self.transformer,  # <--- 注入点：使用携带 FP32 权重的代理
                    image_features=[z, condition_latents[0]],  # z_in 是 JVP 的变量
                    text_features=[prompt_embeds],  # 确保是 float32
                    img_ids=[img_ids, *condition_ids],
                    txt_ids=[text_ids],
                    timesteps=[t_in, t_in] + [torch.zeros_like(t_in)] * len(conditions),
                    delta_t=[delta_t, delta_t] + [torch.zeros_like(delta_t)] * len(conditions),
                    pooled_projections=[pooled_prompt_embeds] * branch_n,  # 确保是 float32
                    guidances=[guidance] * branch_n,
                    adapters=self.adapter_names,
                    return_dict=False,
                    use_kernel=use_kernel,
                    group_mask=group_mask,
                )[0]
            return out

        # 计算 u_theta^{cfg}(z_t, t, t | c)：类别条件输出（r=t，时间间隔为0）
        # u_cfg_cond = u_theta_cfg_fn(x_t, t, t, cond=True)
        # #print(f"[{local_rank}] Base u_cfg_cond mean: {u_cfg_cond.abs().mean().item():.4f}, {u_cfg_cond.shape}")
        # # 计算 u_theta^{cfg}(z_t, t, t)：类别无条件输出（r=t，时间间隔为0）
        # u_cfg_uncond = u_theta_cfg_fn(x_t, t, t, cond=False)
        # #print(f"[{local_rank}] Base u_cfg_uncond mean: {u_cfg_uncond.abs().mean().item():.4f}, {u_cfg_uncond.shape}")
        #
        # # 3. 按论文公式计算 v_t
        # # v_t = ω*(ε - x) + κ*u_cfg_cond + (1-ω-κ)*u_cfg_uncond
        # v_t = self.omega * (x_1 - x_0) + self.kappa * u_cfg_cond + (1 - self.omega - self.kappa) * u_cfg_uncond

        #print(f"[{local_rank}] Base v_t mean: {v_t.abs().mean().item():.4f}, {v_t.shape}")
        #print(f"v_t:{v_t.dtype}")
        # 2. 正确调用 JVP：fn 为可调用函数，输入/切线向量严格对齐 (z, r, t)
        # 注意：JVP 的 fn 必须是 "输入参数→输出" 的可调用对象，不能直接传函数调用结果

        u_out, dudt_ = torch.func.jvp(
            u_theta_pure_f32,  # 封装为可调用 lambda
            (x_t, r, t),  # 输入：(z_t=x_t, r=起始时间, t=当前时间)
            (v_t, torch.zeros_like(r).to(self.flux_pipe.dtype), torch.ones_like(t).to(self.flux_pipe.dtype)) # 论文公式8的 (v, 0, 1)
        )
        # dudt_per_batch_mean = dudt_.flatten(1).mean(dim=1).abs()
        #
        # # 步骤2：打印结果（与方案 1 一致，两种打印方式任选）
        # per_batch_str = ", ".join([f"{x:.4f}" for x in dudt_per_batch_mean.tolist()])
        # print(f"[{local_rank}]use_kernel=true dudt_ : [{per_batch_str}], original shape: {dudt_.shape}")
        # u_out, dudt = torch.func.jvp(
        #     u_theta_cfg_fn,  # 封装为可调用 lambda
        #     (x_t, r, t),  # 输入：(z_t=x_t, r=起始时间, t=当前时间)
        #     (v_t, torch.zeros_like(r).to(self.flux_pipe.dtype), torch.ones_like(t).to(self.flux_pipe.dtype))
        #     # 论文公式8的 (v, 0, 1)
        # )
        # dudt_per_batch_mean = dudt.flatten(1).mean(dim=1).abs()
        #
        # # 步骤2：打印结果（与方案 1 一致，两种打印方式任选）
        # per_batch_str = ", ".join([f"{x:.4f}" for x in dudt_per_batch_mean.tolist()])
        # print(f"[{local_rank}]use_kernel=false dudt : [{per_batch_str}], original shape: {dudt_.shape}")
        # print(f"[{local_rank}] Base u_out mean: {u_out_.abs().mean().item():.4f}, {u_out_.shape}")
        # epsilon = torch.tensor(1e-2, device=x_t.device, dtype=x_t.dtype)
        #
        # # 2. 计算当前点的输出 (基准点)
        u_out_ = u_theta_pure_f32(x_t, r, t, use_kernel=False)
        # print(f"[{local_rank}] Base u_out_ mean: {u_out_.abs().mean().item():.4f}, {u_out_.shape}")
        # #print(f"u_out:{u_out.dtype}")
        # # 3. 准备扰动后的输入
        # # 因为 epsilon 是 BFloat16，这里的加法和乘法结果会保持 BFloat16
        # x_t_perturbed = x_t + epsilon * v_t
        # r_perturbed = r
        # t_perturbed = t + epsilon
        #
        # # 4. 计算扰动后的输出
        # u_perturbed = u_theta_cfg_fn(x_t_perturbed, r_perturbed, t_perturbed, cond=True)
        #
        # # 5. 计算全导数
        # # 所有操作数都是 BFloat16，除法结果也是 BFloat16
        # dudt_= (u_perturbed - u_out_) / epsilon
        #print(f"[{local_rank}] Base dudt_ mean: {dudt_.abs().mean().item():.4f}, {dudt_.shape}")

        # 无需关闭 Flash Attention，直接运行
        # 1. 中心差分步长
        # eps_val = 1e-2
        # epsilon = torch.tensor(eps_val, device=x_t.device, dtype=x_t.dtype)
        #
        # # 2. 节省显存技巧：使用 no_grad 计算两个扰动点
        # with torch.no_grad():
        #     # t + eps
        #     u_plus = u_theta_cfg_fn(x_t + epsilon * v_t, r, t + epsilon, cond=True)
        #     # t - eps
        #     u_minus = u_theta_cfg_fn(x_t - epsilon * v_t, r, t - epsilon, cond=True)
        #
        #     # 3. 转 float32 计算高精度差分
        #     dudt = (u_plus.to(torch.float32) - u_minus.to(torch.float32)) / (2 * eps_val)
        #     dudt = dudt.to(dtype=v_t.dtype)
        #     dudt_per_batch_mean = dudt.flatten(1).mean(dim=1).abs()
        #
        #     # 步骤2：打印结果（与方案 1 一致，两种打印方式任选）
        #     per_batch_str = ", ".join([f"{x:.4f}" for x in dudt_per_batch_mean.tolist()])
        #     print(f"[{local_rank}] Base dudt per batch mean: [{per_batch_str}], original shape: {dudt.shape}")

        # 4. 正常前向传播 (带梯度)
        # print(f"[{local_rank}] Base u_out mean: {u_out_.abs().mean().item():.4f}, {u_out_.shape}")
        # 3. 计算目标平均速度 u_tgt（论文4.1公式10）
        delta_t_expanded = delta_t.unsqueeze(1).unsqueeze(1)  # 适配 latent 维度
        # print(f"[{local_rank}] Base delta_t_expanded mean: {delta_t_expanded.abs().mean().item():.4f}, {delta_t_expanded.shape}")
        u_tgt = v_t - delta_t_expanded * dudt_
        # print(f"[{local_rank}] Base u_tgt mean: {u_tgt.abs().mean().item():.4f}, {u_tgt.shape}")
        #print(f"u_tgt:{u_tgt.dtype}, {u_tgt.shape}")
        # 4. 计算 MSE 损失（论文4.1公式9），对 u_tgt 施加 stop-gradient
        def adaptive_weighted_loss(pred, target, c=1e-3, p=0.5):
            """
            自适应加权 L2 损失函数

            Args:
                pred: 模型预测值 (u_out)
                target: 目标值 (u_tgt)，会自动 detach
                c: 防止除零的小常数，默认 1e-3
                p: 权重指数，p = 1 - γ，默认 1
            Returns:
                加权损失标量
            """
            # 确保常数类型一致
            c = torch.tensor(c, device=pred.device, dtype=pred.dtype)

            # 计算回归误差
            delta = pred - target.detach()

            # L2 平方误差
            l2_squared = delta ** 2

            # 自适应权重（带 stop gradient）
            weight = (1.0 / (l2_squared + c) ** p).detach()
            #print(f"[{local_rank}] weight mean: {weight.abs().mean().item():.4f}, {weight.shape}")
            # 加权损失
            loss = (weight * l2_squared).mean()

            return loss

        # 使用方式
        step_loss = adaptive_weighted_loss(u_out_, u_tgt, c=1e-3, p=1)
        # step_loss = F.mse_loss(u_out, u_tgt.detach(), reduction="mean")

        # -------------------------- 原逻辑保留：日志和返回 --------------------------
        self.last_t = t.mean().item()
        self.last_r = r.mean().item()  # 新增：记录 r 的均值
        self.last_delta_t = delta_t.mean().item()  # 新增：记录时间间隔均值

        self.log_loss = (
            step_loss.item()
            if not hasattr(self, "log_loss")
            else self.log_loss * 0.95 + step_loss.item() * 0.05
        )
        return step_loss

    def generate_a_sample(self):
        raise NotImplementedError("Generate a sample not implemented.")


class TrainingCallback(L.Callback):
    def __init__(self, run_name, training_config: dict = {}, test_function=None):
        self.run_name, self.training_config = run_name, training_config

        self.print_every_n_steps = training_config.get("print_every_n_steps", 10)
        self.save_interval = training_config.get("save_interval", 1000)
        self.sample_interval = training_config.get("sample_interval", 1000)
        self.save_path = training_config.get("save_path", "./output")

        self.wandb_config = training_config.get("wandb", None)
        self.use_wandb = (
            wandb is not None and os.environ.get("WANDB_API_KEY") is not None
        )

        self.total_steps = 0
        self.test_function = test_function

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        gradient_size = 0
        max_gradient_size = 0
        count = 0
        for _, param in pl_module.named_parameters():
            if param.grad is not None:
                gradient_size += param.grad.norm(2).item()
                max_gradient_size = max(max_gradient_size, param.grad.norm(2).item())
                count += 1
        if count > 0:
            gradient_size /= count

        self.total_steps += 1

        # Print training progress every n steps
        if self.use_wandb:
            report_dict = {
                "steps": batch_idx,
                "steps": self.total_steps,
                "epoch": trainer.current_epoch,
                "gradient_size": gradient_size,
            }
            loss_value = outputs["loss"].item() * trainer.accumulate_grad_batches
            report_dict["loss"] = loss_value
            report_dict["t"] = pl_module.last_t
            wandb.log(report_dict)

        if self.total_steps % self.print_every_n_steps == 0:
            print(
                f"Epoch: {trainer.current_epoch}, Steps: {self.total_steps}, Batch: {batch_idx}, Loss: {pl_module.log_loss:.4f}, Gradient size: {gradient_size:.4f}, Max gradient size: {max_gradient_size:.4f}"
            )

        # Save LoRA weights at specified intervals
        if self.total_steps % self.save_interval == 0:
            print(
                f"Epoch: {trainer.current_epoch}, Steps: {self.total_steps} - Saving LoRA weights"
            )
            pl_module.save_lora(
                f"{self.save_path}/{self.run_name}/ckpt/{self.total_steps}"
            )
            pl_module.save_custom_embed_weights(
                f"{self.save_path}/{self.run_name}/ckpt/{self.total_steps}"
            )

        # Generate and save a sample image at specified intervals
        if self.total_steps % self.sample_interval == 0 and self.test_function:
            print(
                f"Epoch: {trainer.current_epoch}, Steps: {self.total_steps} - Generating a sample"
            )
            pl_module.eval()
            self.test_function(
                pl_module,
                f"{self.save_path}/{self.run_name}/output",
                f"lora_{self.total_steps}",
            )
            pl_module.train()


def train(dataset, trainable_model, config, test_function):
    # Initialize
    is_main_process, rank = get_rank() == 0, get_rank()
    torch.cuda.set_device(rank)
    # config = get_config()

    training_config = config["train"]
    run_name = time.strftime("%Y%m%d-%H%M%S")

    # Initialize WanDB
    wandb_config = training_config.get("wandb", None)
    if wandb_config is not None and is_main_process:
        init_wandb(wandb_config, run_name)

    # print("Rank:", rank)
    if is_main_process:
        print("Config:", config)

    # Initialize dataloader
    print("Dataset length:", len(dataset))
    train_loader = DataLoader(
        dataset,
        batch_size=training_config.get("batch_size", 1),
        shuffle=True,
        num_workers=training_config["dataloader_workers"],
    )

    # Callbacks for testing and saving checkpoints
    if is_main_process:
        callbacks = [TrainingCallback(run_name, training_config, test_function)]

    # Initialize trainer
    trainer = L.Trainer(
        # accelerator="cuda",  # 使用 CUDA
        # devices=1,  # 从 Lightning 角度看是 1 个"设备"（但模型内部跨多卡）
        # strategy="auto",
        accumulate_grad_batches=training_config["accumulate_grad_batches"],
        callbacks=callbacks if is_main_process else [],
        enable_checkpointing=False,
        enable_progress_bar=False,
        logger=False,
        max_steps=training_config.get("max_steps", -1),
        max_epochs=training_config.get("max_epochs", -1),
        gradient_clip_val=training_config.get("gradient_clip_val", 0.5),
    )

    setattr(trainer, "training_config", training_config)
    setattr(trainable_model, "training_config", training_config)

    # Save the training config
    save_path = training_config.get("save_path", "./output")
    if is_main_process:
        os.makedirs(f"{save_path}/{run_name}")
        with open(f"{save_path}/{run_name}/config.yaml", "w") as f:
            yaml.dump(config, f)

    # Start training
    trainer.fit(trainable_model, train_loader)
