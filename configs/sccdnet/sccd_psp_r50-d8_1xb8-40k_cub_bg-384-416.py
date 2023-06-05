_base_ = [
    '../_base_/models/sccd_psphead_r50-d8.py', '../_base_/datasets/cub.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]
crop_size = (384, 384)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor, decode_head=dict(num_classes=201),
    auxiliary_head=[
        dict(
            type='FCNHead',
            in_channels=1024,
            in_index=2,
            channels=256,
            num_convs=1,
            concat_input=False,
            dropout_ratio=0.1,
            num_classes=201,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
        dict(
            type='FRCHead',
            in_channels=2048,
            in_index=3,
            channels=512,
            num_classes=201,
            align_corners=False,
            loss_decode=dict(
                type='FRCLoss', use_sigmoid=False, loss_weight=0.01, num_classes=201))
    ])
