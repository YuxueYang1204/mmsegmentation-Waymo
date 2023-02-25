# dataset settings
dataset_type = 'WaymoDataset'
data_root = 'data/waymo/'
pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadNPYAnnotations'),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
data = dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='img_dir/train',
        ann_dir='ann_dir/train',
        pipeline=pipeline)
