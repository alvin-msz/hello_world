import copy
import math
import os
import shutil
from functools import partial

import cv2
import horizon_plugin_pytorch as horizon
import numpy as np
import torch
from horizon_plugin_pytorch.march import March
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor

try:
    from torchvision.transforms.functional_tensor import resize
except ImportError:
    # torchvision 0.18
    from torchvision.transforms._functional_tensor import resize

from horizon_plugin_pytorch.dtype import qint8, qint16
from horizon_plugin_pytorch.quantization import get_qconfig, observer_v2
from horizon_plugin_pytorch.quantization.qconfig_setter import (
    ConvDtypeTemplate,
    MatmulDtypeTemplate,
    ModuleNameTemplate,
    QconfigSetter,
)

from hat.data.collates.nusc_collates import collate_nuscenes
from hat.engine.processors.loss_collector import collect_loss_by_index
from hat.utils.aibenchmark_bpu.bev_process_bpu import (
    preprocess_data,
    process_occ_data,
    process_occ_metric,
)
from hat.utils.checkpoint import update_state_dict_by_add_prefix
from hat.utils.config import ConfigVersion

VERSION = ConfigVersion.v2
training_step = os.environ.get("HAT_TRAINING_STEP", "float")

enable_model_tracking = True

task_name = "flashocc_henet_lss_occ3d_nuscenes_bevfusion_0618_2"
num_classes = 18
batch_size_per_gpu = 8
val_batch_size_per_gpu = 1
ckpt_dir = "./tmp_models/%s" % task_name
cudnn_benchmark = True
seed = None
log_rank_zero_only = True
march = March.NASH_M
qat_mode = "fuse_bn"
convert_mode = "jit-strip"
dataset_type = "NuscenesBevDataset"

data_rootdir = "/open_explorer/data_msz/"

# Data
train_data_path = "data/nuscenes/pack/horizon_bevfusion_nuscenes/train_lmdb/"
val_data_path = "data/nuscenes/pack/horizon_bevfusion_nuscenes/val_lmdb/"
meta_rootdir = "data/nuscenes"
file_client_args = dict(backend='disk')

# LiDAR pretrained checkpoint directory
# lidar_ckpt_dir = "./tmp_pretrained_models/centerpoint_pointpillar_nuscenes"
lidar_ckpt_dir = None

train_interval = 1
val_interval = 1
val_log_interval = 2
num_epochs = 150
# num_epochs=20
step_log_freq = 100
# device_ids = [4, 5, 6, 7]
device_ids = [4,5,6,7]
# device_ids = [0, 1]



def get_feature_map_size(point_cloud_range, voxel_size):
    point_cloud_range = np.array(point_cloud_range, dtype=np.float32)
    voxel_size = np.array(voxel_size, dtype=np.float32)
    grid_size = (point_cloud_range[3:] - point_cloud_range[:3]) / voxel_size
    grid_size = np.round(grid_size).astype(np.int64)
    return grid_size


def get_grid_quant_scale(grid_shape, view_shape):
    max_coord = max(*grid_shape, *view_shape)
    coord_bit_num = math.ceil(math.log(max_coord + 1, 2))
    coord_shift = 15 - coord_bit_num
    coord_shift = max(min(coord_shift, 8), 0)
    grid_quant_scale = 1.0 / (1 << coord_shift)
    return grid_quant_scale


bn_kwargs = dict(eps=2e-5, momentum=0.1)
depth = 45
num_points = 10
bev_size = (40, 40, 0.625)
grid_size = (128, 128)

orig_shape = (3, 900, 1600)
data_shape = (3, 512, 960)
val_data_shape = (3, 512, 960)
resize_shape = (3, 540, 960)

point_cloud_range = [-40, -40, -1, 40, 40, 5.4]
voxel_size = [0.4, 0.4, 6.4]
max_num_points = 20
max_voxels = (30000, 40000)

view_shape = [data_shape[1] / 32, data_shape[2] / 32]
vt_input_hw = [int(view_shape[0]), int(view_shape[1])]
depthview_shape = [6 * depth, view_shape[0] * view_shape[1]]
featview_shape = [view_shape[0] * 6, view_shape[1]]
grid_quant_scale = get_grid_quant_scale(grid_size, featview_shape)
depth_quant_scale = get_grid_quant_scale(grid_size, depthview_shape)

occ3d_seg_class = [
    "others",
    "barrier",
    "bicycle",
    "bus",
    "car",
    "construction_vehicle",
    "motorcycle",
    "pedestrian",
    "traffic_cone",
    "trailer",
    "truck",
    "driveable_surface",
    "other_flat",
    "sidewalk",
    "terrain",
    "manmade",
    "vegetation",
]

lidar_network = dict(
    type="CenterPointDetector",
    feature_map_shape=get_feature_map_size(point_cloud_range, voxel_size),
    pre_process=dict(
        type="CenterPointPreProcess",
        pc_range=point_cloud_range,
        voxel_size=voxel_size,
        max_voxels_num=max_voxels,
        max_points_in_voxel=max_num_points,
        norm_range=[-40, -40, -1, 0, 40, 40, 5.4, 255.0],
        norm_dims=[0, 1, 2, 3],
    ),
    reader=dict(
        type="PillarFeatureNet",
        num_input_features=5,
        num_filters=(64,),
        with_distance=False,
        pool_size=(max_num_points, 1),
        voxel_size=voxel_size,
        pc_range=point_cloud_range,
        bn_kwargs=None,
        quantize=True,
        use_4dim=True,
        use_conv=True,
        hw_reverse=True,
    ),
    backbone=dict(
        type="PointPillarScatter",
        num_input_features=64,
        use_horizon_pillar_scatter=True,
        quantize=True,
    ),
    neck=dict(
        type="SECONDNeck",
        in_feature_channel=64,
        down_layer_nums=[3, 5, 5],
        down_layer_strides=[2, 2, 2],
        down_layer_channels=[64, 128, 256],
        up_layer_strides=[0.5, 1, 2],
        up_layer_channels=[128, 128, 128],
        bn_kwargs=None,
        quantize=True,
        use_relu6=False,
    ),
    # NOTE: detector head (criterion/post_process) not needed for occ task;
    # lidar features are consumed by the fusion module.
)

camera_network = dict(
    type="ViewFusion",
    bev_feat_index=-1,
    bev_upscale=2,
    output_bev_feat=True,
    backbone=dict(
        type="HENet",
        in_channels=3,
        block_nums=[4, 3, 8, 6],
        embed_dims=[64, 128, 192, 384],
        attention_block_num=[0, 0, 0, 0],
        mlp_ratios=[2, 2, 2, 3],
        mlp_ratio_attn=2,
        act_layer=["nn.GELU", "nn.GELU", "nn.GELU", "nn.GELU"],
        use_layer_scale=[True, True, True, True],
        layer_scale_init_value=1e-5,
        num_classes=1000,
        include_top=False,
        extra_act=[False, False, False, False],
        final_expand_channel=0,
        feature_mix_channel=1024,
        block_cls=["GroupDWCB", "GroupDWCB", "AltDWCB", "DWCB"],
        down_cls=["S2DDown", "S2DDown", "S2DDown", "None"],
        patch_embed="origin",
    ),
    neck=dict(
        type="FPN",
        in_strides=[2, 4, 8, 16, 32],
        in_channels=[64, 64, 128, 192, 384],
        out_strides=[16, 32],
        out_channels=[256, 256],
        bn_kwargs=dict(eps=1e-5, momentum=0.1),
    ),
    view_transformer=dict(
        type="LSSTransformer",
        in_channels=256,
        feat_channels=64,
        z_range=(-1.0, 5.4),
        depth=depth,
        num_points=num_points,
        bev_size=bev_size,
        grid_size=grid_size,
        num_views=6,
        grid_quant_scale=grid_quant_scale,
        depth_grid_quant_scale=depth_quant_scale,
        # use_vtv2=True,
        # cal_minmax=False
    ),
    bev_encoder=None,
    bev_decoders=[]
)


model = dict(
    type="BevFusion",
    lidar_network=lidar_network,
    camera_network=camera_network,
    bev_h=256,
    bev_w=256,
    fuse_module=dict(
        type="BevFuseModule",
        input_c=64 + 384,   # LSSTransformer output (64) + SECONDNeck output (384)
        fuse_c=64,         # fused feature dimension for occ head
    ),
    bev_encoder=dict(
        type="BevEncoder",
        backbone=dict(
            type="HENet",
            in_channels=64,
            block_nums=[4, 3, 8, 6],
            embed_dims=[64, 128, 192, 384],
            attention_block_num=[0, 0, 0, 0],
            mlp_ratios=[2, 2, 2, 3],
            mlp_ratio_attn=2,
            act_layer=["nn.GELU", "nn.GELU", "nn.GELU", "nn.GELU"],
            use_layer_scale=[True, True, True, True],
            layer_scale_init_value=1e-5,
            num_classes=1000,
            include_top=False,
            extra_act=[False, False, False, False],
            final_expand_channel=0,
            feature_mix_channel=1024,
            block_cls=["GroupDWCB", "GroupDWCB", "AltDWCB", "DWCB"],
            down_cls=["S2DDown", "S2DDown", "S2DDown", "None"],
            patch_embed="origin",
            quant_input=False,
        ),
        neck=dict(
            type="BiFPN",
            in_strides=[2, 4, 8, 16, 32],
            out_strides=[2, 4, 8, 16, 32],
            stride2channels=dict({2: 64, 4: 64, 8: 128, 16: 192, 32: 384}),
            out_channels=48,
            num_outs=5,
            stack=3,
            start_level=0,
            end_level=-1,
            fpn_name="bifpn_sum",
            upsample_type="function",
            use_fx=True,
        ),
    ),
    bev_decoders=[
        dict(
            type="FlashOccDetDecoder",
            use_mask=True,
            num_classes=num_classes,
            lidar_input=True,
            camera_input=True,
            occ_head=dict(
                type="BEVOCCHead2D",
                in_dim=48,  # matches fuse_module.fuse_c (was 48 for camera-only)
                out_dim=128,
                Dz=16,
                num_classes=num_classes,
                use_predicter=True,
                use_upsample=True,
            ),
            loss_occ=dict(
                type="CrossEntropyLoss",
                use_sigmoid=False,
                ignore_index=255,
                loss_weight=1.0,
            ),
        ),
    ],
)

deploy_model = copy.deepcopy(model)
deploy_model["lidar_network"].pop("pre_process")
deploy_model['camera_network']['compile_model'] = True
deploy_model["bev_decoders"][0]["is_compile"] = True
deploy_model["bev_decoders"][0]["add_argmax_for_compile"] = True
deploy_model["bev_decoders"][0].pop("loss_occ")
deploy_model['bev_decoders']

bda_aug_conf = dict(
    rot_lim=(-0.0, 0.0),
    scale_lim=(1.0, 1.0),
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5,
)

scale = float(resize_shape[2]) / float(orig_shape[2])
resize_aug = (-0.06, 0.11)
train_transforms = [
    dict(type="Lidar2Ego"),
    dict(
        type="MultiViewsImgResize",
        scales=tuple(x + scale for x in resize_aug),
    ),
    dict(type="MultiViewsImgCrop", size=data_shape[1:]),
    dict(type="MultiViewsImgFlip", prob=0.5),
    dict(type="MultiViewsImgRotate", rot=(-5.4, 5.4)),
    dict(
        type="BevFeatureAug",
        bda_aug_conf=bda_aug_conf,
        is_train=True,
    ),
    dict(
        type="MultiViewsImgTransformWrapper",
        transforms=[
            dict(type="PILToTensor"),
            dict(type="BgrToYuv444", rgb_input=True),
            dict(type="Normalize", mean=128.0, std=128.0),
        ],
    ),
]

test_transforms = [
    dict(type="Lidar2Ego"),
    dict(type="MultiViewsImgResize", size=resize_shape[1:]),
    dict(type="MultiViewsImgCrop", size=data_shape[1:]),
    dict(
        type="MultiViewsImgTransformWrapper",
        transforms=[
            dict(type="PILToTensor"),
            dict(type="BgrToYuv444", rgb_input=True),
            dict(type="Normalize", mean=128.0, std=128.0),
        ],
    ),
]

bup_transforms = [
    dict(type="MultiViewsImgResize", size=resize_shape[1:]),
    dict(type="MultiViewsImgCrop", size=data_shape[1:]),
]

hbm_transforms = [
    dict(type="MultiViewsImgResize", size=resize_shape[1:]),
    dict(type="MultiViewsImgCrop", size=data_shape[1:]),
    dict(
        type="MultiViewsImgTransformWrapper",
        transforms=[
            dict(type="PILToTensor"),
        ],
    ),
]

data = dict(
    train=dict(
        type=dataset_type,
        data_path=os.path.join(data_rootdir, "train_lmdb"),
        bev_size=bev_size,
        with_bev_mask=False,
        with_ego_occ=True,
        # with_lidar_occ=True,
        need_lidar=True,
        num_sweeps=0,
        load_dim=5,
        use_dim=[0, 1, 2, 3, 4],
        transforms=train_transforms,
    ),
    val=dict(
        type=dataset_type,
        data_path=os.path.join(data_rootdir, "val_lmdb"),
        bev_size=bev_size,
        with_bev_mask=False,
        with_ego_occ=True,
        need_lidar=True,
        # with_lidar_occ=True,
        num_sweeps=0,
        load_dim=5,
        use_dim=[0, 1, 2, 3, 4],
        transforms=test_transforms,
    ),
    bup_val=dict(
        type=dataset_type,
        data_path=os.path.join(data_rootdir, "val_lmdb"),
        bev_size=bev_size,
        with_bev_mask=False,
        with_ego_occ=True,
        # with_lidar_occ=True,
        need_lidar=True,
        transforms=bup_transforms,
    ),
    hbm_val=dict(
        type=dataset_type,
        data_path=os.path.join(data_rootdir, "val_lmdb"),
        bev_size=bev_size,
        with_bev_mask=False,
        with_ego_occ=True,
        # with_lidar_occ=True,
        need_lidar=True,
        transforms=hbm_transforms,
    ),
)

bpu_preprocess_data = preprocess_data
bpu_val_dataset = data["bup_val"]

data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=data["train"],
    sampler=dict(type=torch.utils.data.DistributedSampler),
    batch_size=batch_size_per_gpu,
    shuffle=False,
    num_workers=5,
    collate_fn=collate_nuscenes,
    pin_memory=True,
)

val_data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=data["val"],
    sampler=dict(type=torch.utils.data.DistributedSampler),
    batch_size=val_batch_size_per_gpu,
    shuffle=False,
    num_workers=5,
    collate_fn=collate_nuscenes,
    pin_memory=True,
)


batch_processor = dict(
    type="BasicBatchProcessor",
    need_grad_update=True,
    loss_collector=collect_loss_by_index(1),
    enable_amp=True,
    enable_amp_dtype=torch.float16,
)

val_batch_processor = dict(
    type="BasicBatchProcessor",
    need_grad_update=False,
    loss_collector=None,
)


def update_loss(metrics, batch, model_out):
    for metric in metrics:
        metric.update(model_out[1])


def val_update_metric_func(metrics, batch, model_outs):
    gt_semantics = batch["gt_occ_info"]["voxel_semantics"][
        0
    ].squeeze()  # (Dx, Dy, Dz)

    # For BEVFusion: use union of lidar and camera masks
    lidar_mask = batch["gt_occ_info"]["mask_lidar"][0].squeeze()
    camera_mask = batch["gt_occ_info"]["mask_camera"][0].squeeze()
    # mask = lidar_mask | camera_mask
    mask = camera_mask

    # print(model_outs[0].shape)
    if isinstance(model_outs, list):
        model_outs = model_outs[0]
    semantics_pred = model_outs[1]["occ_pre"].squeeze()  # (Dx, Dy, Dz)
    # semantics_pred = model_outs[0].squeeze().argmax(dim=-1)  # (Dx, Dy)
    masked_semantics_gt = gt_semantics[mask]
    masked_semantics_pred = semantics_pred[mask]

    results = {
        "label": masked_semantics_gt.reshape(-1),
        "preds": masked_semantics_pred.reshape(-1),
    }

    for metric in metrics:
        metric.update(**results)


def val_update_metric_func_add_argmax(metrics, batch, model_outs):
    """Variant for ``add_argmax_for_compile=True`` in FlashOccDetDecoder.

    Handles BOTH raw-logit and already-compressed model outputs by checking
    the tensor rank:

    * 5-D (B, Dx, Dy, Dz, C) → raw logits → applies argmax(dim=-1)
    * 4-D (B, Dx, Dy, Dz)     → already compressed → used as-is
    * 3-D (Dx, Dy, Dz)         → compressed, no batch → used as-is

    Supports BevFusion (list-wrapped), ViewFusion (bare tuple), and
    calibration (raw tensor) output formats.  For batch size > 1 only the
    first sample is evaluated, matching the GT extraction logic.
    """
    gt_semantics = batch["gt_occ_info"]["voxel_semantics"][
        0
    ].squeeze()  # (Dx, Dy, Dz)

    lidar_mask = batch["gt_occ_info"]["mask_lidar"][0].squeeze()
    camera_mask = batch["gt_occ_info"]["mask_camera"][0].squeeze()
    # mask = lidar_mask | camera_mask
    mask = camera_mask.to(torch.bool)

    # BevFusion: model_outs = [( [occ_preds], {"occ_pre": broken} )]
    # ViewFusion: model_outs = ( [occ_preds], {"occ_pre": broken} )
    if isinstance(model_outs, list):
        model_outs = model_outs[0]

    # model_outs[0] is normally ``[occ_preds]`` (list of one tensor).
    # During calibration it may be the raw tensor directly.
    raw = model_outs[0]
    if isinstance(raw, (list, tuple)):
        raw = raw[0]

    # 5-D  = raw logits  (B, Dx, Dy, Dz, num_classes) → need argmax
    # <=4-D = already argmax-compressed  (B, Dx, Dy, Dz) or (Dx, Dy, Dz)
    if raw.dim() >= 5:
        raw = raw.argmax(dim=-1)

    # Drop batch dimension to match gt_semantics: (B, Dx, Dy, Dz) → (Dx, Dy, Dz)
    while raw.dim() > 3:
        raw = raw[0]

    semantics_pred = raw.squeeze().long()  # (Dx, Dy, Dz)

    masked_semantics_gt = gt_semantics[mask]
    masked_semantics_pred = semantics_pred[mask]

    results = {
        "label": masked_semantics_gt.reshape(-1).long(),
        "preds": masked_semantics_pred.reshape(-1).long(),
    }

    for metric in metrics:
        metric.update(**results)

metric_updater = dict(
    type="MetricUpdater",
    metric_update_func=update_loss,
    step_log_freq=step_log_freq,
    epoch_log_freq=1,
    log_prefix=task_name,
)

val_metric_updater = dict(
    type="MetricUpdater",
    metric_update_func=val_update_metric_func,
    step_log_freq=1000000,
    epoch_log_freq=1,
    log_prefix="Validation " + task_name,
)

val_metric_updater_add_argmax = dict(
    type="MetricUpdater",
    metric_update_func=val_update_metric_func_add_argmax,
    step_log_freq=1000000,
    epoch_log_freq=1,
    log_prefix="Validation " + task_name,
)

stat_callback = dict(
    type="StatsMonitor",
    log_freq=500,
)

ckpt_callback = dict(
    type="Checkpoint",
    save_dir=ckpt_dir,
    name_prefix=training_step + "-",
    strict_match=True,
    mode="max",
)

grad_callback = dict(
    type="GradScale",
    module_and_scale=[],
    clip_grad_norm=35,
    clip_norm_type=2,
)

val_callback = dict(
    type="Validation",
    data_loader=val_data_loader,
    val_interval=val_log_interval,
    batch_processor=val_batch_processor,
    callbacks=[val_metric_updater],
    val_model=model,
    val_on_train_end=True,
)


val_callback_argmax = dict(
    type="Validation",
    data_loader=val_data_loader,
    val_interval=val_log_interval,
    batch_processor=val_batch_processor,
    callbacks=[val_metric_updater_add_argmax],
    val_model=model,
    val_on_train_end=True,
)


float_trainer = dict(
    type="distributed_data_parallel_trainer",
    model=model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        converters=[
            # # Load camera branch pretrained weights (HENet ImageNet pretrained)
            # dict(
            #     type="LoadCheckpoint",
            #     checkpoint_path="/data01/chenmu/bev_lss_v2/weights/henet/float-checkpoint-best.pth.tar",
            #     allow_miss=True,
            #     ignore_extra=True,
            #     ignore_tensor_shape=True,
            # ),
            # # Load lidar branch pretrained weights with prefix
            # dict(
            #     type="LoadCheckpoint",
            #     checkpoint_path=os.path.join(
            #         lidar_ckpt_dir, "float-checkpoint-last.pth.tar"
            #     ),
            #     state_dict_update_func=partial(
            #         update_state_dict_by_add_prefix, prefix="lidar_net."
            #     ),
            #     allow_miss=True,
            #     ignore_extra=True,
            #     verbose=True,
            # ),
            dict(
                type='LoadCheckpoint',
                checkpoint_path='/data01/chenmu/bev_lss_v2/release_package/scripts/tmp_models/flashocc_henet_lss_occ3d_nuscenes_bevfusion_0618/float-checkpoint-best.pth.tar',
                allow_miss=True,
                ignore_extra=True,
                ignore_tensor_shape=True,
                verbose=True
            )
        ],
    ),
    data_loader=data_loader,
    optimizer=dict(
        type=torch.optim.AdamW,
        # lr=1e-4 * 0.5,
        lr = 1e-4,
        weight_decay=1e-2,
    ),
    batch_processor=batch_processor,
    num_epochs=num_epochs,
    device=None,
    resume_optimizer=False,
    resume_epoch_or_step=False,
    resume_dataloader=False,
    callbacks=[
        stat_callback,
        grad_callback,
        dict(
            type="CosineAnnealingLrUpdater",
            warmup_len=500,
            warmup_by="step",
            warmup_lr_ratio=1.0 / 3,
            step_log_interval=500,
            stop_lr=1e-5
        ),
        metric_updater,
        val_callback,
        ckpt_callback,
    ],
    sync_bn=True,
    train_metrics=dict(
        type="LossShow",
    ),
    val_metrics=dict(
        type="MeanIOU",
        seg_class=occ3d_seg_class,
        ignore_index=17,
    ),
)

bpu_eval_metric = dict(
    type="MeanIOU",
    seg_class=occ3d_seg_class,
    ignore_index=17,
)
bpu_process_metric = process_occ_metric
bpu_process_result_data = process_occ_data

# Enable argmax before dequant during calibration so the calibrated model
# matches compile-time behaviour (argmax → int32 before DeQuantStub).
calib_model = copy.deepcopy(model)
calib_model["bev_decoders"][0]["add_argmax_for_compile"] = True

calibration_data_loader = copy.deepcopy(data_loader)
calibration_data_loader.pop("sampler")
calibration_batch_processor = copy.deepcopy(val_batch_processor)
# calibration_val_callback = copy.deepcopy(val_callback)
calibration_val_callback = val_callback_argmax
calibration_val_callback["val_interval"] = 1

calibration_data_loader["batch_size"] = batch_size_per_gpu
calibration_val_callback["val_on_train_end"] = False
calibration_step = 100

calibration_example_data_loader = copy.deepcopy(calibration_data_loader)
calibration_example_data_loader["num_workers"] = 0

val_example_data_loader = copy.deepcopy(val_data_loader)
val_example_data_loader.pop("sampler")
val_example_data_loader["num_workers"] = 0

int16_qconfig = dict()
int16_qconfig["view_transformer.quant_stub"] = {"dtype": qint16}
int16_qconfig["view_transformer.dquant_stub"] = {"dtype": qint16}

qtemplates = [
    ModuleNameTemplate({"": qint8}),
    ModuleNameTemplate(
        int16_qconfig,
        freeze=True,
    ),
    MatmulDtypeTemplate(
        input_dtypes=[qint8, qint8],
    ),
    ConvDtypeTemplate(
        input_dtype=qint8,
        weight_dtype=qint8,
    ),
]

cali_qconfig_setter = QconfigSetter(
    reference_qconfig=get_qconfig(
        observer=(observer_v2.MSEObserver)
    ),
    templates=qtemplates,
    enable_optimize=True,
    save_dir=ckpt_dir,
    custom_qconfig_mapping=None,
)

calibration_val_callback["model_convert_pipeline"] = dict(
    type="ModelConvertPipeline",
    qat_mode="fuse_bn",
    converters=[
        dict(
            type="Float2Calibration",
            convert_mode=convert_mode,
            example_data_loader=val_example_data_loader,
            qconfig_setter=cali_qconfig_setter,
        ),
    ],
)
calibration_ckpt_callback = copy.deepcopy(ckpt_callback)
calibration_ckpt_callback["save_interval"] = 1


def get_example_inputs(num_points):
    inputs = {"img": torch.randn((6,) + data_shape)}
    inputs["ego2img"] = torch.randn(
        (6, 4, 4),
    )
    N = 35000
    pts = torch.zeros(N, 5)
    pts[:, 0] = torch.rand(N) * (point_cloud_range[3] - point_cloud_range[0]) + point_cloud_range[0]
    pts[:, 1] = torch.rand(N) * (point_cloud_range[4] - point_cloud_range[1]) + point_cloud_range[1]
    pts[:, 2] = torch.rand(N) * (point_cloud_range[5] - point_cloud_range[2]) + point_cloud_range[2]
    pts[:, 3] = torch.rand(N)
    pts[:, 4] = 0  # time_delta (0 for current sweep)
    inputs["points"] = [pts]  # batch_size=1 for tracing
    return inputs


example_inputs = get_example_inputs(num_points)

calibration_trainer = dict(
    type="Calibrator",
    model=calib_model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode="fuse_bn",
        converters=[
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    ckpt_dir, "float-checkpoint-best.pth.tar"
                ),
            ),
            dict(
                type="Float2Calibration",
                convert_mode=convert_mode,
                example_inputs=example_inputs,
                qconfig_setter=cali_qconfig_setter,
            ),
            dict(
                type="FixWeightQScale",
            ),
        ],
    ),
    data_loader=calibration_data_loader,
    batch_processor=calibration_batch_processor,
    num_steps=calibration_step,
    device=None,
    callbacks=[
        stat_callback,
        calibration_val_callback,
        calibration_ckpt_callback,
    ],
    val_metrics=dict(
        type="MeanIOU",
        seg_class=occ3d_seg_class,
        ignore_index=17,
    ),
    log_interval=calibration_step / 10,
)


# predictor
float_predictor = dict(
    type="Predictor",
    model=model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        converters=[
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    ckpt_dir, "float-checkpoint-last.pth.tar"
                ),
            ),
        ],
    ),
    data_loader=[val_data_loader],
    batch_processor=val_batch_processor,
    device=None,
    metrics=dict(
        type="MeanIOU",
        seg_class=occ3d_seg_class,
        ignore_index=17,
    ),
    callbacks=[
        val_metric_updater,
    ],
    log_interval=1,
)

calibration_predictor = dict(
    type="Predictor",
    model=model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode="fuse_bn",
        converters=[
            dict(
                type="Float2QAT",
                convert_mode=convert_mode,
                example_inputs=example_inputs,
                qconfig_setter=cali_qconfig_setter,
            ),
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    ckpt_dir, "calibration-checkpoint-best.pth.tar"
                ),
                ignore_extra=True,
            ),
        ],
    ),
    data_loader=[val_data_loader],
    batch_processor=val_batch_processor,
    device=None,
    metrics=dict(
        type="MeanIOU",
        seg_class=occ3d_seg_class,
        ignore_index=17,
    ),
    callbacks=[
        # val_metric_updater,
        val_metric_updater_add_argmax
    ],
    log_interval=1,
)


hbir_deploy_model = copy.deepcopy(deploy_model)

hbir_infer_model = dict(
    type="BevFusionHbirInfer",
    deploy_model=hbir_deploy_model,
    ir_model=dict(
        type="HbirModule",
        model_path=os.path.join(ckpt_dir, "quantized.bc"),
    ),
)

int_infer_data_loader = copy.deepcopy(val_data_loader)
int_infer_data_loader["batch_size"] = 1
int_infer_data_loader["shuffle"] = False

int_infer_predictor = dict(
    type="Predictor",
    model=hbir_infer_model,
    data_loader=int_infer_data_loader,
    batch_processor=val_batch_processor,
    device=None,
    metrics=dict(
        type="MeanIOU",
        seg_class=occ3d_seg_class,
        ignore_index=17,
    ),
    callbacks=[
        val_metric_updater,
    ],
    log_interval=1,
)
align_bpu_predictor = dict(
    type="Predictor",
    model=hbir_infer_model,
    data_loader=int_infer_data_loader,
    batch_processor=val_batch_processor,
    device=None,
    metrics=dict(
        type="MeanIOU",
        seg_class=occ3d_seg_class,
        ignore_index=17,
    ),
    callbacks=[
        val_metric_updater,
    ],
    log_interval=1,
)

# Deploy inputs for BevFusion: voxelized lidar features + coords + camera images
deploy_inputs = {
    "features": torch.randn(
        (1, 5, max_num_points, max_voxels[0]), dtype=torch.float32,
    ),
    "coors": torch.zeros([max_voxels[0], 4], dtype=torch.int32),
    "img": torch.randn((6, 3, data_shape[1], data_shape[2])),
    "ego2img": torch.randn((6, 4, 4)),
    # pre-computed reference points for compile_model=True
    # shape: (num_points, grid_h, grid_w, 2)
    "points0": torch.randn((num_points, grid_size[0], grid_size[1], 2)),
    "points1": torch.randn((num_points, grid_size[0], grid_size[1], 2)),
}

deploy_model_convert_pipeline = dict(
    type="ModelConvertPipeline",
    qat_mode="fuse_bn",
    converters=[
        dict(
            type="Float2QAT",
            convert_mode=convert_mode,
            example_inputs=deploy_inputs,
            qconfig_setter=cali_qconfig_setter,
        ),
        dict(
            type="LoadCheckpoint",
            checkpoint_path=os.path.join(
                ckpt_dir, "calibration-checkpoint-best.pth.tar"
            ),
            ignore_extra=True,
            allow_miss=True,
            verbose=True,
        ),
    ],
)

hbir_exporter = dict(
    type="HbirExporter",
    model=deploy_model,
    model_convert_pipeline=deploy_model_convert_pipeline,
    example_inputs=deploy_inputs,
    save_path=ckpt_dir,
    model_name=task_name,
    input_names=list(deploy_inputs.keys()),
    enable_vpu=False,
)

# compile_dir = os.path.join(ckpt_dir, "compile")
# compile_cfg = dict(
#     march=march,
#     name=task_name + "_model",
#     hbm=os.path.join(compile_dir, "model.hbm"),
#     layer_details=True,
#     input_source="pyramid,ddr,ddr",
#     opt="O2",
#     split_dim=dict(
#         inputs={
#             "0": [0, 6],
#         }
#     ),
# )
compile_dir = os.path.join(ckpt_dir, "compile")
compile_cfg = dict(
    march=march,
    name=task_name,
    hbm=os.path.join(compile_dir, "model.hbm"),
    layer_details=True,
    input_source=[
        "ddr",
    ],
    opt="O2",
)


def resize_homo(homo, scale):
    view = np.eye(4)
    view[0, 0] = scale[1]
    view[1, 1] = scale[0]
    homo = view @ homo
    return homo


def crop_homo(homo, offset):
    view = np.eye(4)
    view[0, 2] = -offset[0]
    view[1, 2] = -offset[1]
    homo = view @ homo
    return homo


def process_img(img_path, resize_size, crop_size):
    orig_img = cv2.imread(img_path)
    cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB, orig_img)
    orig_img = Image.fromarray(orig_img)
    orig_img = pil_to_tensor(orig_img)
    resize_hw = (
        int(resize_size[0]),
        int(resize_size[1]),
    )

    orig_shape = (orig_img.shape[1], orig_img.shape[2])
    resized_img = resize(orig_img, resize_hw).unsqueeze(0)
    top = int(resize_hw[0] - crop_size[0])
    left = int((resize_hw[1] - crop_size[1]) / 2)
    resized_img = resized_img[:, :, top:, left:]

    return resized_img, orig_shape


def process_inputs(infer_inputs, transforms=None):

    resize_size = resize_shape[1:]
    input_size = val_data_shape[1:]
    orig_imgs = []
    file_list = list(os.listdir(infer_inputs))
    image_dir_list = list(filter(lambda x: x.endswith(".jpg"), file_list))
    image_dir_list.sort()
    for i, img in enumerate(image_dir_list):
        img = os.path.join(infer_inputs, img)
        img, orig_shape = process_img(img, resize_size, input_size)
        orig_imgs.append({"name": i, "img": img})

    input_imgs = []
    for orig_img in orig_imgs:
        input_img = horizon.nn.functional.bgr_to_yuv444(orig_img["img"], True)
        input_imgs.append(input_img)

    input_imgs = torch.cat(input_imgs)
    input_imgs = (input_imgs - 128.0) / 128.0

    homo = np.load(os.path.join(infer_inputs, "ego2img.npy")).astype("float64")

    top = int(resize_size[0] - input_size[0])
    left = int((resize_size[1] - input_size[1]) / 2)

    scale = (resize_size[0] / orig_shape[0], resize_size[1] / orig_shape[1])
    homo = resize_homo(homo, scale)
    homo = crop_homo(homo, (left, top))

    model_input = {
        "img": input_imgs,
        "ego2img": torch.tensor(homo),
    }
    if transforms is not None:
        model_input = transforms(model_input)

    return model_input, None


def process_outputs(model_outs, viz_func, vis_inputs):
    semantics_pred = (
        model_outs[1]["occ_pre"].squeeze().numpy().astype(np.uint8)
    )

    viz_func(semantics_pred)
    return None


single_infer_dataset = copy.deepcopy(int_infer_data_loader["dataset"])
single_infer_dataset["transforms"] = None


def inputs_save_func(data, save_path):
    if os.path.isdir(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path, exist_ok=True)
    for image_idx, img_data in enumerate(data["img"]):
        save_name = f"img{image_idx}.jpg"
        img_data.save(os.path.join(save_path, save_name), "JPEG")

    ego2img_path = os.path.join(save_path, "ego2img.npy")
    np.save(ego2img_path, np.array(data["ego2img"]))


infer_cfg = dict(
    model=hbir_infer_model,
    input_path=f"./demo/{task_name}",
    gen_inputs_cfg=dict(
        dataset=single_infer_dataset,
        sample_idx=[0],
        inputs_save_func=inputs_save_func,
    ),
    process_inputs=process_inputs,
    viz_func=dict(
        type="OccViz",
        vcs_range=(-40.0, -40.0, -1.0, 40.0, 40.0, 5.4),
        vis_bev_2d=True,
    ),
    process_outputs=process_outputs,
)

onnx_cfg = dict(
    model=deploy_model,
    inputs=deploy_inputs,
    stage="qat",
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode="fuse_bn",
        converters=[
            dict(
                type="Float2QAT",
                convert_mode=convert_mode,
                example_inputs=deploy_inputs,
                qconfig_setter=cali_qconfig_setter,
            ),
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    ckpt_dir, "calibration-checkpoint-best.pth.tar"
                ),
            ),
        ],
    ),
)
