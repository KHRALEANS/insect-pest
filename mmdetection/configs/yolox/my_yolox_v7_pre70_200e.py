_base_ = ['../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py']

img_scale = (1280, 1280)  # height, width

# model settings
model = dict(
    type='YOLOX',
    input_size=img_scale,
    random_size_range=(15, 25),
    random_size_interval=10,
    backbone=dict(type='CSPDarknet', deepen_factor=0.33, widen_factor=0.5),
    neck=dict(
        type='YOLOXPAFPN',
        in_channels=[128, 256, 512],
        out_channels=128,
        num_csp_blocks=1),
    bbox_head=dict(
        type='YOLOXHead', num_classes=29, in_channels=128, feat_channels=128),
    train_cfg=dict(assigner=dict(type='SimOTAAssigner', center_radius=2.5)),
    # In order to align the source code, the threshold of the val phase is
    # 0.01, and the threshold of the test phase is 0.001.
    test_cfg=dict(score_thr=0.01, nms=dict(type='nms', iou_threshold=0.65)))

# dataset settings
data_root = 'data/pest-coco/AUG/'
dataset_type = 'CocoDataset'
classes = ('无', '二化螟', '二点委夜蛾', '棉铃虫', '褐飞虱属', '白背飞虱', '八点灰灯蛾', '蝼蛄', '蟋蟀', '甜菜夜蛾', '黄足猎蝽', '稻纵卷叶螟', '甜菜白带野螟', '黄毒蛾', '石蛾', '大黑鳃金龟', '粘虫', '稻螟蛉', '甘蓝夜蛾', '地老虎', '大螟', '瓜绢野螟', '线委夜蛾', '水螟蛾', '紫条尺蛾', '歧角螟', '草地螟', '豆野螟', '干纹冬夜蛾')


train_pipeline = [
    # added for test
    # dict(type='LoadImageFromFile'),
    # added for test
    # dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Mosaic', img_scale=img_scale, pad_val=114.0),
    dict(
        type='RandomAffine',
        scaling_ratio_range=(0.1, 2),
        border=(-img_scale[0] // 2, -img_scale[1] // 2)),
    dict(
        type='MixUp',
        img_scale=img_scale,
        ratio_range=(0.8, 1.6),
        pad_val=114.0),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', flip_ratio=0.5),
    # According to the official implementation, multi-scale
    # training is not considered here but in the
    # 'mmdet/models/detectors/yolox.py'.
    dict(type='Resize', img_scale=img_scale, keep_ratio=True),
    dict(
        type='Pad',
        pad_to_square=True,
        # If the image is three-channel, the pad value needs
        # to be set separately for each channel.
        pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]

train_dataset = dict(
    type='MultiImageMixDataset',
    dataset=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'annotations/train_labels.json',
        img_prefix=data_root + 'train/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True)
        ],
        filter_empty_gt=False,
    ),
    pipeline=train_pipeline)

test_pipeline = [
    dict(type='LoadImageFromFile'),
    # added for test
    # dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='MultiScaleFlipAug',
        img_scale=img_scale,
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Pad',
                pad_to_square=True,
                pad_val=dict(img=(114.0, 114.0, 114.0))),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ])
]

data = dict(
    samples_per_gpu=10,
    workers_per_gpu=8,
    persistent_workers=True,
    train=train_dataset,
    val=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'annotations/val_labels.json',
        img_prefix=data_root + 'val/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'annotations/val_labels.json',
        img_prefix=data_root + 'val/',
        pipeline=test_pipeline))

# optimizer
# default 8 gpu
# 4 gpu Tesla P40
optimizer = dict(
    type='SGD',
    lr=0.00625,
    momentum=0.9,
    weight_decay=5e-4,
    nesterov=True,
    paramwise_cfg=dict(norm_decay_mult=0., bias_decay_mult=0.))
optimizer_config = dict(grad_clip=None)

max_epochs = 250
num_last_epochs = 15
resume_from = None
load_from = 'work_dirs/my_yolox_v6_pre60_250e/epoch_70.pth'
interval = 10

# learning policy
lr_config = dict(
    _delete_=True,
    policy='YOLOX',
    warmup='exp',
    by_epoch=False,
    warmup_by_epoch=True,
    warmup_ratio=1,
    warmup_iters=5,  # 5 epoch
    num_last_epochs=num_last_epochs,
    min_lr_ratio=0.05)

runner = dict(type='EpochBasedRunner', max_epochs=max_epochs)

custom_hooks = [
    dict(
        type='YOLOXModeSwitchHook',
        num_last_epochs=num_last_epochs,
        priority=48),
    dict(
        type='SyncNormHook',
        num_last_epochs=num_last_epochs,
        interval=interval,
        priority=48),
    dict(
        type='ExpMomentumEMAHook',
        resume_from=resume_from,
        momentum=0.0001,
        priority=49)
]
checkpoint_config = dict(interval=interval)
evaluation = dict(
    save_best='auto',
    # The evaluation interval is 'interval' when running epoch is
    # less than ‘max_epochs - num_last_epochs’.
    # The evaluation interval is 1 when running epoch is greater than
    # or equal to ‘max_epochs - num_last_epochs’.
    interval=1,
    dynamic_intervals=[(max_epochs - num_last_epochs, 1)],
    metric='bbox')
log_config = dict(interval=5)

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (8 samples per GPU)
auto_scale_lr = dict(base_batch_size=64)