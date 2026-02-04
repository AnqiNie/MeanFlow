export PYTHONPATH=/workspace/nieanqi/OminiControl:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=4
# *[Specify the config file path]
export OMINI_CONFIG=./train/config/spatial_alignment.yaml
echo $OMINI_CONFIG
export TOKENIZERS_PARALLELISM=true

python -m test.sample