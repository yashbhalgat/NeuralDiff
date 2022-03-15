
VID=$1; CUDA_VISIBLE_DEVICES=0 python train.py \
  --vid $VID \
  --train_ratio 1.0 --num_epochs 10 --use_hash
