num_classes = 5
norm_cfg = dict(type='BN', requires_grad=True)
backbone_norm_cfg = dict(type='LN', requires_grad=True)
class_weight = [0.7, 1.2, 1.2, 1, 1]
model = dict(
    type='EncoderDecoder',
    # 预训练模型路径 这是官方的预训练模型 只包含backbone的权重 latest.pth是自己训练时保存的模型检查点，包含了完整的模型权重
    # 首次训练时使用 pretrained 加载官方预训练模型初始化backbone
    # 后续训练时使用 load_from 加载自己训练的模型继续优化
    pretrained=
    'IML_code/checkpoint/swin_base_patch4_window7_mmseg.pth',
    backbone=dict(
        type='SwinTransformer',
        pretrain_img_size=224,
        embed_dims=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=7,
        use_abs_pos_embed=False,
        drop_path_rate=0.3,
        patch_norm=True,
        patch_size=4,
        mlp_ratio=4,
        strides=(4, 2, 2, 2),
        out_indices=(0, 1, 2, 3),
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        act_cfg=dict(type='GELU'),
        norm_cfg=dict(type='LN', requires_grad=True)),
    decode_head=dict(
        type='UPerHead',
        in_channels=[128, 256, 512, 1024],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=5,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False,
        loss_decode=[
            dict(
                type='CrossEntropyLoss',
                use_sigmoid=False,
                loss_weight=1.0,
                class_weight=[0.7, 1.2, 1.2, 1, 1]),
            dict(
                type='LovaszLoss',
                loss_weight=1.0,
                reduction='none',
                class_weight=[0.7, 1.2, 1.2, 1, 1])
        ],
        sampler=dict(type='OHEMPixelSampler', thresh=0.7, min_kept=100000)),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=512,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=5,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False,
        loss_decode=[
            dict(
                type='CrossEntropyLoss',
                use_sigmoid=False,
                loss_weight=1.0,
                class_weight=[0.7, 1.2, 1.2, 1, 1]),
            dict(type='LovaszLoss', loss_weight=1.0, reduction='none')
        ],
        sampler=dict(type='OHEMPixelSampler', thresh=0.7, min_kept=100000)),
    train_cfg=dict(),
    test_cfg=dict(mode='slide', stride=(512, 512), crop_size=(512, 512)))
dataset_type = 'CustomDataset'
data_root = 'data/datasetF/'
classes = ['auth', 'copymove', 'splice', 'text', 'erase']
palette = [[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255], [180, 0, 200]]
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
size = 512
crop_size = 512
ratio = 1.0
albu_train_transforms = [dict(type='ColorJitter', p=0.5)]
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='RandomCrop', cat_max_ratio=0.75, crop_size=(512, 512)),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    dict(type='PhotoMetricDistortion'),
    # dict(type='Albu', transforms=[dict(type='ColorJitter', p=0.5)]),
    dict(type='Resize', img_scale=(512, 512)),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size=(512, 512), pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=None,
        img_ratios=[1.0],
        flip=False,
        flip_direction=['horizontal', 'vertical'],
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='ResizeToMultiple', size_divisor=32),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=1,
    train=dict(
        type='CustomDataset',
        data_root='/data/yxj/data/datasetF/',
        img_dir='dataTamperF24/MixE/img',
        ann_dir='dataTamperF24/MixE/maskC',
        img_suffix='.png',
        seg_map_suffix='.png',
        classes=['auth', 'copymove', 'splice', 'text', 'erase'],
        palette=[[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255],
                 [180, 0, 200]],
        # use_mosaic=False,
        # mosaic_prob=0.5,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(type='RandomCrop', cat_max_ratio=0.75, crop_size=(512, 512)),
            dict(type='RandomFlip', prob=0.5, direction='horizontal'),
            dict(type='RandomFlip', prob=0.5, direction='vertical'),
            dict(type='PhotoMetricDistortion'),
            # dict(type='Albu', transforms=[dict(type='ColorJitter', p=0.5)]),
            dict(type='Resize', img_scale=(512, 512)),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size=(512, 512), pad_val=0, seg_pad_val=255),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_semantic_seg'])
        ]),
    val=dict(
        type='CustomDataset',
        data_root='/data/yxj/data/datasetF/',
        img_dir='test/MixE/img',
        ann_dir='test/MixE/maskC',
        test_mode=True,
        img_suffix='.png',
        seg_map_suffix='.png',
        classes=['auth', 'copymove', 'splice', 'text', 'erase'],
        palette=[[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255],
                 [180, 0, 200]],
        # use_mosaic=False,
        # mosaic_prob=0.5,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=None,
                img_ratios=[1.0],
                flip=False,
                flip_direction=['horizontal', 'vertical'],
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='ResizeToMultiple', size_divisor=32),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='CustomDataset',
        data_root='/data/yxj/data/',
        img_dir='datasetF/code2/img',
        ann_dir='datasetF/code2/maskC',
        test_mode=True,
        img_suffix='.png',
        seg_map_suffix='.png',
        classes=['auth', 'copymove', 'splice', 'text', 'erase'],
        palette=[[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255],
                 [180, 0, 200]],
        # use_mosaic=False,
        # mosaic_prob=0.5,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=None,
                img_ratios=[1.0],
                flip=False,
                flip_direction=['horizontal', 'vertical'],
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='ResizeToMultiple', size_divisor=32),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
# 每50次迭代输出一次训练日志 
log_config = dict(
    interval=50, hooks=[dict(type='TextLoggerHook', by_epoch=False)])
dist_params = dict(backend='nccl')
log_level = 'INFO'

# 从指定路径加载预训练模型 不会加载优化器状态、学习率调度器状态、训练轮次等训练状态
# load_from = 'IML_code/train_checkpoint/1103/latest.pth'
load_from = None
# load_from = 'swin-t/Swin-Transformer-Semantic-Segmentation/work_config/240506/mixE/latest.pth'
resume_from = None

workflow = [('train', 1)]
cudnn_benchmark = True
nx = 6
total_epochs = 4
optimizer = dict(
    constructor='LearningRateDecayOptimizerConstructor',
    type='AdamW',
    lr=0.0001,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        decay_rate=0.99, decay_type='stage_wise', num_layers=12))
optimizer_config = dict(type='Fp16OptimizerHook', loss_scale=512.0)
lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-06,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)
runner = dict(type='EpochBasedRunner', max_epochs=24)
checkpoint_config = dict(by_epoch=True, interval=1, save_optimizer=False)
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend')
]
visualizer = dict(
    type='SegLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend')
    ],
    name='visualizer')
evaluation = dict(
    by_epoch=True, interval=1, metric=['mIoU', 'mFscore'], pre_eval=True)
fp16 = dict(loss_scale=512.0)
work_dir = 'IML_code/train_checkpoint/1103'
device = 'cuda'
gpu_ids = [0]
