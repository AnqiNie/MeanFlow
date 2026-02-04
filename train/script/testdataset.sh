export PYTHONPATH=/workspace/nieanqi/OminiControl:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=4
export OMINI_CONFIG=./train/config/spatial_alignment.yaml
echo $OMINI_CONFIG


python -m test.testdataset