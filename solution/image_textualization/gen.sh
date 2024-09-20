#!/usr/bin/bash

CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"

conda activate vllm  # 切换到自己的conda环境

# 设置 CUDA 可见设备并运行 Python 脚本，使用vllm加速
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python ./extract/extract_fr_desc-llama.py \
    --input_file data/first_input.jsonl \
    --output_file data/obj_extr_from_desc.jsonl \
    --stop_tokens "<|eot_id|>" \
    --prompt_structure "<|begin_of_text|><|start_header_id|>user<|end_header_id|>{input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>" \
    --start_line 0 \
    --end_line 50000

# 使用 sed 命令替换文本
sed -i "s/\n\n%%%RESPONSE%%%//g" data/obj_extr_from_desc.jsonl

conda activate it

# step2.1 从图像中获取obj
CUDA_VISIBLE_DEVICES=0,1 python extract/extract_fr_img.py \
    --test_task DenseCap \
    --config_file ./extract/configs/GRiT_B_DenseCap_ObjectDet.yaml \
    --confidence_threshold 0.55 \
    --image_folder  /home/disk2/zhangjingjun/data_syn/dj_synth_challenge/input/pretrain_stage_1/images \
    --input_file  data/first_input_without_desc.jsonl \
    --output_file data/obj_extr_from_img.jsonl \
    --start_line 0 \
    --end_line 50000 \
    --opts MODEL.WEIGHTS ./ckpt/grit_b_densecap_objectdet.pth

# step3 在step2的基础上运行，只支持单卡
CUDA_VISIBLE_DEVICES=2 python fg_annotation/mask_depth.py \
    --input_path data/obj_extr_from_img.jsonl \
    --output_path data/fg_anno_ddp.jsonl \
    --image_folder  /home/disk2/zhangjingjun/data_syn/dj_synth_challenge/input/pretrain_stage_1/images \
    --image_depth_folder data/depth_map_folder \
    --start_line 0 \
    --end_line 50000

# step2.2 目前在cpu上跑，速度较快
CUDA_VISIBLE_DEVICES=0,1 python filter/filter_fr_desc.py \
    --model_config ./filter/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
    --model_checkpoint ./ckpt/groundingdino_swinb_ogc.pth \
    --box_threshold 0.20 \
    --text_threshold 0.18 \
    --input_file data/obj_extr_from_desc.jsonl \
    --output_file data/hal_from_desc.jsonl \
    --image_folder  /home/disk2/zhangjingjun/data_syn/dj_synth_challenge/input/pretrain_stage_1/images \
    --start_line 0 \
    --end_line 50000
# step2.3 支持多卡，但速度较慢
CUDA_VISIBLE_DEVICES=6,7 python ./utils/trans_img2depth.py \
    --input_file data/first_input_without_desc.jsonl \
    --output_folder data/depth_map_folder \
    --image_folder   /home/disk2/zhangjingjun/data_syn/dj_synth_challenge/input/pretrain_stage_1/images \
    --start_line 0 \
    --end_line 50000

# step4
python ./data/merge.py

# step5
conda activate vllm
# 使用vllm加速
CUDA_VISIBLE_DEVICES=0,1,6,7 python ./refine/modify.py \
    --input_file data/your.jsonl \
    --output_file data/refined_desc.jsonl \
    --stop_tokens "<|eot_id|>" \
    --prompt_structure "<|begin_of_text|><|start_header_id|>user<|end_header_id|>{input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>" \
    --start_line 0 \
    --end_line 50000

