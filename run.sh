CONFIG=configs/convnext/upernet_convnext_base_fp16_1024x1024_160k_cityscapes.py
OUT=work_dirs/convnext_1024_1024_160
bash tools/dist_train.sh $CONFIG 8 --work-dir $OUT --seed 0

# Waymo Inference
# bash tools/dist_infer.sh $CONFIG work_dirs/segformer_b5/latest.pth 8 --work-dir $OUT --format-only --eval-options imgfile_prefix=work_dirs/pc_mask