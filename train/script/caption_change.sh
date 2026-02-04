export PYTHONPATH=/workspace/nieanqi/OminiControl:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=4,5
export OMINI_CONFIG=./train/config/spatial_alignment.yaml
echo $OMINI_CONFIG
accelerate launch --num_processes 2 -m dataset.caption_change \
  --dataset_path cache/t2i2m \
  --output_path caption \
  --batch_size 32 \
  --replacement_ratio 0.6 \
  --verification_samples 100 \
  --verification_dir vertify