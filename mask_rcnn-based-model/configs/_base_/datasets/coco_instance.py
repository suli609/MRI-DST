dataset_type = 'CocoDataset'
#data_root = './data_wenhekoulou/coco/'
data_root = './data/coco/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
albu_train_transforms = [
    dict(
        type='ShiftScaleRotate',
        shift_limit=0.0625,
        scale_limit=0.0,
        rotate_limit=0,
        interpolation=1,
        p=0.5),
    dict(
        type='RandomBrightnessContrast',
        brightness_limit=[0.1, 0.3],
        contrast_limit=[0.1, 0.3],
        p=0.2),
#    dict(
#        type='OneOf',
#        transforms=[
#            dict(
#                type='RGBShift',
#                r_shift_limit=10,
#                g_shift_limit=10,
#                b_shift_limit=10,
#                p=1.0),
#            dict(
#                type='HueSaturationValue',
#                hue_shift_limit=20,
#                sat_shift_limit=30,
#                val_shift_limit=20,
#                p=1.0)
#        ],
#        p=0.1),
    dict(type='JpegCompression', quality_lower=85, quality_upper=95, p=0.2),
#    dict(type='ChannelShuffle', p=0.1),
    dict(
        type='OneOf',
        transforms=[
            dict(type='Blur', blur_limit=3, p=1.0),
            dict(type='MedianBlur', blur_limit=3, p=1.0)
        ],
        p=0.1),
]
train_pipeline = [
    dict(type='LoadImageFromFile',to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
#    dict(type='Albu', transforms = [{"type": 'RandomRotate90'}]),
#    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
#    dict(type='Resize', img_scale=(768, 768), keep_ratio=True),
    dict(type='Resize', img_scale=(912, 608), keep_ratio=True),
#    dict(
#        type='Albu',
#        transforms=albu_train_transforms,
#        bbox_params=dict(
#            type='BboxParams',
#            format='pascal_voc',
        #    format='coco',
#            label_fields=['gt_labels'],
#            min_visibility=0.0,
#            filter_lost_elements=True),
#        keymap={
#            'img': 'image',
#            'gt_masks': 'masks',
#            'gt_bboxes': 'bboxes'
#        },
#        update_pad_shape=False,
#        skip_img_without_anno=True),
    dict(type='MinIoURandomCrop'),
    dict(type='PhotoMetricDistortion'),
    dict(type='Corrupt',corruption='jpeg_compression'),
    dict(type='Corrupt',corruption='gaussian_blur'),
#    dict(type='Rotate1', prob=0.6,level=10,max_rotate_angle=60),
    dict(type='RandomFlip', flip_ratio=0.5,direction='horizontal'),
    dict(type='RandomFlip', flip_ratio=0.5,direction='vertical'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
#    dict(
#        type='Collect',
#        keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'],
#        meta_keys=('filename', 'ori_shape', 'img_shape', 'img_norm_cfg',
#                   'pad_shape', 'scale_factor'))
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
#        img_scale=(1333, 800),
#        img_scale=(768, 768),
        img_scale=(912, 608),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
#    samples_per_gpu=2,
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_train2017.json',
    #    img_prefix=data_root + 'images/train2017/',
        img_prefix=data_root + 'train2017/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
    #    img_prefix=data_root + 'images/val2017/',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_test2017.json',
    #    img_prefix=data_root + 'images/val2017/',
        img_prefix=data_root + 'test2017/',
        pipeline=test_pipeline))
evaluation = dict(interval=5,metric=['bbox', 'segm'])
