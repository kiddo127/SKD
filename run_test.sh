# peeled vit_tiny
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port 29506 --use_env main.py \
    --eval \
    --model vit_tiny --batch-size 256 \
    --data-path /path/to/ImageNet2012/Data/CLS-LOC \
    --output_dir ./output/peeled/vit_tiny \
    --training_peeled --op_code 64 \
    --peelable_resume ./output/peelable/vit_tiny/checkpoint_299.pth \
    --mask_table_path ./output/peelable/vit_tiny/mask_table.json

# 39 (70.0)   MACs: 897.0 MMac, Params: 4.77 M
# 64 (69.0)   MACs: 792.69 MMac, Params: 4.25 M
# 66 (69.0)   MACs: 785.4 MMac, Params: 4.21 M
# 84 (68.6)   MACs: 732.37 MMac, Params: 3.94 M
# 90 (68.2)   MACs: 712.28 MMac, Params: 3.84 M
# 99 (67.8)   MACs: 692.17 MMac, Params: 3.74 M
# 100 (67.8)   MACs: 688.51 MMac, Params: 3.72 M
# 111 (67.3)   MACs: 659.25 MMac, Params: 3.57 M
# 130    MACs: 611.74 MMac, Params: 3.33 M
# 144 (65.8)   MACs: 555.11 MMac, Params: 3.04 M
# 150    MACs: 522.23 MMac, Params: 2.88 M
# 156 (65.1)   MACs: 509.44 MMac, Params: 2.81 M
# 200 (61.4)   MACs: 379.69 MMac, Params: 2.16 M
# 222 (1/4)   MACs: 277.21 MMac, Params: 1.64 M


# peeled vit_small
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port 29506 --use_env main.py \
    --eval \
    --model vit_small --batch-size 256 \
    --data-path /path/to/ImageNet2012/Data/CLS-LOC \
    --output_dir ./output/peeled/vit_small \
    --training_peeled --op_code 49 \
    --peelable_resume ./output/peelable/vit_small/checkpoint_299.pth \
    --mask_table_path ./output/peelable/vit_small/mask_table.json

# 248 (1/4)   MACs: 1.01 GMac, Params: 5.58 M
# 264 (1/5)   MACs: 847.67 MMac, Params: 4.76 M
# 214 (70.6)   MACs: 1.45 GMac, Params: 7.83 M
# 160 (76.2)   MACs: 2.15 GMac, Params: 11.34 M
# 150 (76.7)   MACs: 2.27 GMac, Params: 11.97 M
# 134 (77.3)   MACs: 2.43 GMac, Params: 12.78 M
# 95 (78.2)   MACs: 2.82 GMac, Params: 14.74 M
# 70 (78.1)   MACs: 3.07 GMac, Params: 16.03 M
# 49 (79.0)   MACs: 3.57 GMac, Params: 18.54 M


# peeled vit_base
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port 29506 --use_env main.py \
    --eval \
    --model vit_base --batch-size 256 \
    --data-path /path/to/ImageNet2012/Data/CLS-LOC \
    --output_dir ./output/peeled/vit_base \
    --training_peeled --op_code 70 \
    --peelable_resume ./output/peelable/vit_base/checkpoint_149.pth \
    --mask_table_path ./output/peelable/vit_base/mask_table.json

# 70 (81.5)  MACs: 13.91 GMac, Params: 71.52 M
# 80 (81.4)  MACs: 13.42 GMac, Params: 69.01 M
# 127 (81.1)  MACs: 11.3 GMac, Params: 58.23 M
# 130 (80.9)  MACs: 11.03 GMac, Params: 56.91 M
# 140 (81.0)  MACs: 10.57 GMac, Params: 54.55 M
# 150 (80.9)  MACs: 10.1 GMac, Params: 52.18 M
# 194 (80.4)  MACs: 8.47 GMac, Params: 43.92 M
# 210 (80.1)  MACs: 7.89 GMac, Params: 40.96 M
# 268 (77.0)  MACs: 5.65 GMac, Params: 29.6 M
# 298 (1/4)   MACs: 4.2 GMac, Params: 22.22 M






# # original vit_tiny
# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port 29502 --use_env main.py \
#     --eval \
#     --model vit_tiny --batch-size 256 \
#     --data-path /path/to/ImageNet2012/Data/CLS-LOC \
#     --output_dir ./output/test \
#     --num_workers 32 \


# # original vit_small
# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port 29502 --use_env main.py \
#     --eval \
#     --model vit_small --batch-size 256 \
#     --data-path /path/to/ImageNet2012/Data/CLS-LOC \
#     --output_dir ./output/test/vit_small \
#     --num_workers 32 \


# # original vit_base
# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port 29502 --use_env main.py \
#     --eval \
#     --model vit_base --batch-size 256 \
#     --data-path /path/to/ImageNet2012/Data/CLS-LOC \
#     --output_dir ./output/test/vit_base \
#     --num_workers 32 \
