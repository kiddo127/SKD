# peelable vit_tiny
CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --nproc_per_node=2 --master_port 29501 --use_env main.py \
    --training_peelable \
    --model vit_tiny --batch-size 128 \
    --data-path /path/to/ImageNet2012/Data/CLS-LOC \
    --output_dir ./output/peelable/vit_tiny \
    --epochs 300 --warmup-epochs 0 --eval_period 1 \
    --transformed_weight ./output/reparamed/vit_tiny/reparamed_proxySize=1024.pth


# peelable vit_small
CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --nproc_per_node=2 --master_port 29501 --use_env main.py \
    --training_peelable \
    --model vit_small --batch-size 128 \
    --data-path /path/to/ImageNet2012/Data/CLS-LOC \
    --output_dir ./output/peelable/vit_small \
    --epochs 300 --warmup-epochs 0 --eval_period 1 \
    --transformed_weight ./output/reparamed/vit_small/reparamed_proxySize=1024.pth


# peelable vit_base
CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --nproc_per_node=2 --master_port 29501 --use_env main.py \
    --training_peelable \
    --model vit_base --batch-size 128 \
    --data-path /path/to/ImageNet2012/Data/CLS-LOC \
    --output_dir ./output/peelable/vit_base \
    --epochs 150 --warmup-epochs 0 --eval_period 1 \
    --transformed_weight ./output/reparamed/vit_base/reparamed_proxySize=1024.pth \
