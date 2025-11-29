# vit_tiny
CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --nproc_per_node=2 --master_port 29500 --use_env main.py \
    --eval \
    --model vit_tiny --batch-size 128 \
    --data-path /data/huangxin/dataset/ImageNet2012/Data/CLS-LOC \
    --output_dir ./output/reparamed/vit_tiny \
    --reparameterization --proxy_samples 1024 \


# vit_small
CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --nproc_per_node=2 --master_port 29500 --use_env main.py \
    --eval \
    --model vit_small --batch-size 128 \
    --data-path /data/huangxin/dataset/ImageNet2012/Data/CLS-LOC \
    --output_dir ./output/reparamed/vit_small \
    --reparameterization --proxy_samples 1024 \


# vit_base
CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --nproc_per_node=2 --master_port 29500 --use_env main.py \
    --eval \
    --model vit_base --batch-size 128 \
    --data-path /data/huangxin/dataset/ImageNet2012/Data/CLS-LOC \
    --output_dir ./output/reparamed/vit_base \
    --num_workers 32 \
    --reparameterization --proxy_samples 1024 \
