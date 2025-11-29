
# peeled vit_tiny
CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --nproc_per_node=2 --master_port 29503 --use_env main.py \
    --training_peeled --op_code 64 \
    --output_dir ./output/peeled/vit_tiny/op64 \
    --model vit_tiny --batch-size 128 \
    --data-path /path/to/ImageNet2012/Data/CLS-LOC \
    --epochs 30 --warmup-epochs 0 --eval_period 1 \
    --lr 5e-5 --min-lr 1e-6 \
    --peelable_resume ./output/peelable/vit_tiny/checkpoint_299.pth \
    --mask_table_path ./output/peelable/vit_tiny/mask_table.json


# peeled vit_small
CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --nproc_per_node=2 --master_port 29503 --use_env main.py \
    --training_peeled --op_code 150 \
    --output_dir ./output/peeled/vit_small/op150 \
    --model vit_small --batch-size 128 \
    --data-path /path/to/ImageNet2012/Data/CLS-LOC \
    --epochs 30 --warmup-epochs 0 --eval_period 1 \
    --lr 5e-5 --min-lr 1e-6 \
    --peelable_resume ./output/peelable/vit_small/checkpoint_299.pth \
    --mask_table_path ./output/peelable/vit_small/mask_table.json


# peeled vit_base
CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --nproc_per_node=2 --master_port 29503 --use_env main.py \
    --training_peeled --op_code 194 \
    --output_dir ./output/peeled/vit_base/op194 \
    --model vit_base --batch-size 128 \
    --data-path /path/to/ImageNet2012/Data/CLS-LOC \
    --epochs 30 --warmup-epochs 0 --eval_period 1 \
    --lr 5e-5 --min-lr 1e-6 \
    --peelable_resume ./output/peelable/vit_base/checkpoint_149.pth \
    --mask_table_path ./output/peelable/vit_base/mask_table.json
