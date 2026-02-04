# *[Specify the config file path and the GPU devices to use]
export CUDA_VISIBLE_DEVICES=4,5

# *[Specify the config file path]
export OMINI_CONFIG=./train/config/spatial_alignment.yaml

# *[Specify the WANDB API key]
export WANDB_API_KEY='6e59b297698f6d4a53f67e34ea97b3b3f81c8e47'

echo $OMINI_CONFIG
export TOKENIZERS_PARALLELISM=false
## 配置 HF 镜像（临时生效，终端执行）
#export HF_ENDPOINT=https://hf-mirror.com

accelerate launch --main_process_port 41356  --mixed_precision fp16 -m omini.train_flux.train_spatial_alignment
#python -m omini.train_flux.train_spatial_alignment