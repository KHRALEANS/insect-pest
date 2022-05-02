optimizer = dict(
    type='SGD',
    lr=0.00625,
    momentum=0.9,
    weight_decay=0.0005,
    nesterov=True,
    paramwise_cfg=dict(norm_decay_mult=0.0, bias_decay_mult=0.0))
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='YOLOX',
    warmup='exp',
    by_epoch=False,
    warmup_by_epoch=True,
    warmup_ratio=1,
    warmup_iters=5,
    num_last_epochs=15,
    min_lr_ratio=0.05)
runner = dict(type='EpochBasedRunner', max_epochs=250)
checkpoint_config = dict(interval=10)
log_config = dict(interval=5, hooks=[dict(type='TextLoggerHook')])
custom_hooks = [
    dict(type='YOLOXModeSwitchHook', num_last_epochs=15, priority=48),
    dict(type='SyncNormHook', num_last_epochs=15, interval=10, priority=48),
    dict(
        type='ExpMomentumEMAHook',
        resume_from=None,
        momentum=0.0001,
        priority=49)
]
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = 'work_dirs/my_yolox_v6_pre60_250e/epoch_70.pth'
resume_from = None
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
auto_scale_lr = dict(enable=False, base_batch_size=64)
img_scale = (1280, 1280)
model = dict(
    type='YOLOX',
    input_size=(1280, 1280),
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
    test_cfg=dict(score_thr=0.01, nms=dict(type='nms', iou_threshold=0.65)))
data_root = 'data/pest-coco/AUG/'
dataset_type = 'CocoDataset'
classes = ('无', '二化螟', '二点委夜蛾', '棉铃虫', '褐飞虱属', '白背飞虱', '八点灰灯蛾', '蝼蛄', '蟋蟀',
           '甜菜夜蛾', '黄足猎蝽', '稻纵卷叶螟', '甜菜白带野螟', '黄毒蛾', '石蛾', '大黑鳃金龟', '粘虫',
           '稻螟蛉', '甘蓝夜蛾', '地老虎', '大螟', '瓜绢野螟', '线委夜蛾', '水螟蛾', '紫条尺蛾', '歧角螟',
           '草地螟', '豆野螟', '干纹冬夜蛾')
train_pipeline = [
    dict(type='Mosaic', img_scale=(1280, 1280), pad_val=114.0),
    dict(
        type='RandomAffine', scaling_ratio_range=(0.1, 2),
        border=(-640, -640)),
    dict(
        type='MixUp',
        img_scale=(1280, 1280),
        ratio_range=(0.8, 1.6),
        pad_val=114.0),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Resize', img_scale=(1280, 1280), keep_ratio=True),
    dict(
        type='Pad',
        pad_to_square=True,
        pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
train_dataset = dict(
    type='MultiImageMixDataset',
    dataset=dict(
        type='CocoDataset',
        classes=('无', '二化螟', '二点委夜蛾', '棉铃虫', '褐飞虱属', '白背飞虱', '八点灰灯蛾', '蝼蛄',
                 '蟋蟀', '甜菜夜蛾', '黄足猎蝽', '稻纵卷叶螟', '甜菜白带野螟', '黄毒蛾', '石蛾', '大黑鳃金龟',
                 '粘虫', '稻螟蛉', '甘蓝夜蛾', '地老虎', '大螟', '瓜绢野螟', '线委夜蛾', '水螟蛾',
                 '紫条尺蛾', '歧角螟', '草地螟', '豆野螟', '干纹冬夜蛾'),
        ann_file='data/pest-coco/AUG/annotations/train_labels.json',
        img_prefix='data/pest-coco/AUG/train/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True)
        ],
        filter_empty_gt=False),
    pipeline=[
        dict(type='Mosaic', img_scale=(1280, 1280), pad_val=114.0),
        dict(
            type='RandomAffine',
            scaling_ratio_range=(0.1, 2),
            border=(-640, -640)),
        dict(
            type='MixUp',
            img_scale=(1280, 1280),
            ratio_range=(0.8, 1.6),
            pad_val=114.0),
        dict(type='YOLOXHSVRandomAug'),
        dict(type='RandomFlip', flip_ratio=0.5),
        dict(type='Resize', img_scale=(1280, 1280), keep_ratio=True),
        dict(
            type='Pad',
            pad_to_square=True,
            pad_val=dict(img=(114.0, 114.0, 114.0))),
        dict(
            type='FilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
    ])
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1280, 1280),
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
    train=dict(
        type='MultiImageMixDataset',
        dataset=dict(
            type='CocoDataset',
            classes=('无', '二化螟', '二点委夜蛾', '棉铃虫', '褐飞虱属', '白背飞虱', '八点灰灯蛾', '蝼蛄',
                     '蟋蟀', '甜菜夜蛾', '黄足猎蝽', '稻纵卷叶螟', '甜菜白带野螟', '黄毒蛾', '石蛾',
                     '大黑鳃金龟', '粘虫', '稻螟蛉', '甘蓝夜蛾', '地老虎', '大螟', '瓜绢野螟', '线委夜蛾',
                     '水螟蛾', '紫条尺蛾', '歧角螟', '草地螟', '豆野螟', '干纹冬夜蛾'),
            ann_file='data/pest-coco/AUG/annotations/train_labels.json',
            img_prefix='data/pest-coco/AUG/train/',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True)
            ],
            filter_empty_gt=False),
        pipeline=[
            dict(type='Mosaic', img_scale=(1280, 1280), pad_val=114.0),
            dict(
                type='RandomAffine',
                scaling_ratio_range=(0.1, 2),
                border=(-640, -640)),
            dict(
                type='MixUp',
                img_scale=(1280, 1280),
                ratio_range=(0.8, 1.6),
                pad_val=114.0),
            dict(type='YOLOXHSVRandomAug'),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(type='Resize', img_scale=(1280, 1280), keep_ratio=True),
            dict(
                type='Pad',
                pad_to_square=True,
                pad_val=dict(img=(114.0, 114.0, 114.0))),
            dict(
                type='FilterAnnotations',
                min_gt_bbox_wh=(1, 1),
                keep_empty=False),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
        ]),
    val=dict(
        type='CocoDataset',
        classes=('无', '二化螟', '二点委夜蛾', '棉铃虫', '褐飞虱属', '白背飞虱', '八点灰灯蛾', '蝼蛄',
                 '蟋蟀', '甜菜夜蛾', '黄足猎蝽', '稻纵卷叶螟', '甜菜白带野螟', '黄毒蛾', '石蛾', '大黑鳃金龟',
                 '粘虫', '稻螟蛉', '甘蓝夜蛾', '地老虎', '大螟', '瓜绢野螟', '线委夜蛾', '水螟蛾',
                 '紫条尺蛾', '歧角螟', '草地螟', '豆野螟', '干纹冬夜蛾'),
        ann_file='data/pest-coco/AUG/annotations/val_labels.json',
        img_prefix='data/pest-coco/AUG/val/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1280, 1280),
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
        ]),
    test=dict(
        type='CocoDataset',
        classes=('无', '二化螟', '二点委夜蛾', '棉铃虫', '褐飞虱属', '白背飞虱', '八点灰灯蛾', '蝼蛄',
                 '蟋蟀', '甜菜夜蛾', '黄足猎蝽', '稻纵卷叶螟', '甜菜白带野螟', '黄毒蛾', '石蛾', '大黑鳃金龟',
                 '粘虫', '稻螟蛉', '甘蓝夜蛾', '地老虎', '大螟', '瓜绢野螟', '线委夜蛾', '水螟蛾',
                 '紫条尺蛾', '歧角螟', '草地螟', '豆野螟', '干纹冬夜蛾'),
        ann_file='data/pest-coco/AUG/annotations/val_labels.json',
        img_prefix='data/pest-coco/AUG/val/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1280, 1280),
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
        ]))
max_epochs = 250
num_last_epochs = 15
interval = 10
evaluation = dict(
    save_best='auto', interval=1, dynamic_intervals=[(235, 1)], metric='bbox')
work_dir = './work_dirs/my_yolox_v7_pre70_200e'
auto_resume = False
gpu_ids = range(0, 4)
