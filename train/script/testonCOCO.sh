export PYTHONPATH=/workspace/nieanqi/OminiControl:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=5
# *[Specify the config file path]
export OMINI_CONFIG=./train/config/spatial_alignment.yaml
echo $OMINI_CONFIG
export TOKENIZERS_PARALLELISM=true

HF_ENDPOINT=https://hf-mirror.com python -m test.testonCOCO