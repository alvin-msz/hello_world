"""Lidar-only PointPillar + BEVFormerDetDecoder (vs CenterPointHead).

Based on ``centerpoint_pointpillar_nuscenes.py`` (same voxelization / reader /
backbone / SECONDNeck). Detection head replaced by ``BEVFormerDetDecoder`` from
``bevfusion_pointpillar_henet_multisensor_multitask_nuscenes_argmax_cpu.py``.

``CenterPointDetector`` cannot host ``BEVFormerDetDecoder`` (API mismatch), so
the model is ``BevFusion`` with ``camera_network=None`` and a BevFuse-style
384→256 lidar BEV projection (``LidarBEVFormerDetDecoder``).

Training recipe follows DETR/BEVFormer practice in this repo
(``bevfusion_centerpoint_*``, ``bevformer_tiny_*``): CosineAnnealing + warmup,
differential LR (lidar_net 0.1x / decoder 3x), AMP, 24 epochs, and lidar
geometric augs (flip/scale/rot) instead of CenterPoint CyclicLr + CBGS.
"""

import copy
import os
import shutil
import sys
from functools import partial

import numpy as np
import torch
from horizon_plugin_pytorch.quantization import March

from hat.data.collates.nusc_collates import collate_nuscenes_sequencev2
from hat.utils.config import ConfigVersion
from hat.visualize.lidar_det import lidar_det_visualize

_tools_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "tools")
)
if _tools_dir not in sys.path:
    sys.path.insert(0, _tools_dir)
import lidar_bevformer_det_adapter  # noqa: F401
from lidar_bevformer_det_adapter import remap_lidar_centerpoint_state_dict

VERSION = ConfigVersion.v2
training_step = os.environ.get("HAT_TRAINING_STEP", "float")

task_name = "bevformer_pointpillar_nuscenes"
batch_size_per_gpu = 16
device_ids = [0, 1, 2, 3]

ckpt_dir = f"./tmp_models/{task_name}"
# Warm-start PointPillar reader/backbone/neck from CenterPoint float ckpt
lidar_pretrain_ckpt = (
    "./tmp_models/bevformer_pointpillar_nuscenes/"
    "float-checkpoint-last.pth.tar"
)

# datadir settings — align with centerpoint_pointpillar_nuscenes.py
train_data_path = "data/nuscenes/pack/horizon_bevfusion_nuscenes/train_lmdb/"
val_data_path = "data/nuscenes/pack/horizon_bevfusion_nuscenes/val_lmdb/"
meta_rootdir = "data/nuscenes"
log_loss_show = 200

cudnn_benchmark = True
seed = None
log_rank_zero_only = True
march = March.NASH_E
norm_cfg = None
qat_mode = "fuse_bn"
convert_mode = "fx"
enable_vpu = True

# Voxelization cfg (same as CenterPoint)
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_size = [0.2, 0.2, 8]
max_num_points = 20
max_voxels = (30000, 40000)

# BEVFormerDetDecoder settings (from bevfusion argmax_cpu)
_dim_ = 256
bev_h_ = 128
bev_w_ = 128
lidar_bev_channels = 384  # sum(SECONDNeck up_layer_channels)
num_classes = 10
input_size = (512, 960)

# BevFormer / NuScenes CLASSES order (must match BevFormerCriterion / metric)
class_names = (
    "car",
    "truck",
    "trailer",
    "bus",
    "construction_vehicle",
    "bicycle",
    "motorcycle",
    "pedestrian",
    "traffic_cone",
    "barrier",
)


def get_feature_map_size(point_cloud_range, voxel_size):
    point_cloud_range = np.array(point_cloud_range, dtype=np.float32)
    voxel_size = np.array(voxel_size, dtype=np.float32)
    grid_size = (point_cloud_range[3:] - point_cloud_range[:3]) / voxel_size
    grid_size = np.round(grid_size).astype(np.int64)
    return grid_size


# Lidar backbone only (no CenterPointHead / targets / loss / postprocess)
lidar_network = dict(
    type="CenterPointDetector",
    feature_map_shape=get_feature_map_size(point_cloud_range, voxel_size),
    pre_process=dict(
        type="CenterPointPreProcess",
        pc_range=point_cloud_range,
        voxel_size=voxel_size,
        max_voxels_num=max_voxels,
        max_points_in_voxel=max_num_points,
        norm_range=[-51.2, -51.2, -5.0, 0.0, 51.2, 51.2, 3.0, 255.0],
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
        bn_kwargs=norm_cfg,
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
        bn_kwargs=norm_cfg,
        quantize=True,
        use_relu6=False,
    ),
)

# Same BEVFormerDetDecoder block as bevfusion_*_argmax_cpu.py
_bevformer_decoder = dict(
    type="BEVFormerDetDecoder",
    bev_h=bev_h_,
    bev_w=bev_w_,
    num_query=900,
    embed_dims=_dim_,
    pc_range=point_cloud_range,
    decoder=dict(
        type="DetectionTransformerDecoder",
        num_layers=6,
        return_intermediate=True,
        decoder_layer=dict(
            type="DetrTransformerDecoderLayer",
            crossattention=dict(
                type="HorizonMultiScaleDeformableAttention",
                embed_dims=_dim_,
                num_levels=1,
                grid_align_num=4,
                feats_size=[[bev_w_, bev_h_]],
            ),
            dropout=0.1,
        ),
    ),
    criterion=dict(
        type="BevFormerCriterion",
        assigner=dict(
            type="BevFormerHungarianAssigner3D",
            cls_cost=dict(type="FocalLossCost", weight=2.0),
            reg_cost=dict(type="BBox3DL1Cost", weight=0.25),
        ),
        loss_cls=dict(
            type="FocalLoss",
            loss_name="cls",
            num_classes=num_classes + 1,
            alpha=0.25,
            gamma=2.0,
            loss_weight=2.0,
            reduction="mean",
        ),
        loss_bbox=dict(
            type="L1Loss",
            loss_weight=0.25,
        ),
        pc_range=point_cloud_range,
        bbox_key="lidar_bboxes_labels",
    ),
    post_process=dict(
        type="BevFormerProcess",
        post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
        pc_range=point_cloud_range,
        max_num=300,
        num_classes=num_classes,
    ),
)

bev_head = dict(
    type="LidarBEVFormerDetDecoder",
    in_channels=lidar_bev_channels,
    out_channels=_dim_,
    # BevFuse-style proj: SECOND 384ch -> DETR-friendly 256ch BEV
    use_fuse_proj=True,
    decoder=_bevformer_decoder,
)

model = dict(
    type="BevFusion",
    lidar_network=lidar_network,
    camera_network=None,
    bev_decoders=[bev_head],
    fuse_module=None,
    bev_h=bev_h_,
    bev_w=bev_w_,
)

deploy_model = copy.deepcopy(model)
deploy_model["lidar_network"].pop("pre_process")
deploy_model["bev_decoders"][0]["is_compile"] = True
deploy_model["bev_decoders"][0]["decoder"]["is_compile"] = True
deploy_model["bev_decoders"][0]["decoder"].pop("criterion", None)
deploy_model["bev_decoders"][0]["decoder"].pop("post_process", None)

deploy_inputs = dict(
    features=torch.randn((1, 5, 20, 40000), dtype=torch.float32),
    coors=torch.zeros([40000, 4]).int(),
)

# BevFormerCriterion needs seq_meta / lidar_bboxes_labels.
# Lidar geometric augs (flip/scale/rot) matter far more than image augs here.
train_dataset = dict(
    type="NuscenesBevSequenceDataset",
    data_path=train_data_path,
    map_size=None,
    map_path=None,
    with_bev_mask=False,
    with_lidar_bboxes=True,
    with_bev_bboxes=False,
    with_ego_bboxes=True,
    bev_range=point_cloud_range,
    need_lidar=True,
    num_sweeps=1,
    load_dim=5,
    use_dim=[0, 1, 2, 3, 4],
    num_seq=1,
    with_ego_occ=False,
    with_lidar_occ=False,
    transforms=[
        # mini image pipeline (dataset still packs cams; model ignores img)
        dict(type="MultiViewsImgResize", size=input_size),
        dict(
            type="MultiViewsImgTransformWrapper",
            transforms=[
                dict(type="PILToTensor"),
                dict(type="BgrToYuv444", rgb_input=True),
                dict(type="Normalize", mean=128, std=128),
            ],
        ),
        # CenterPoint-like lidar augs for query DETR head
        dict(
            type="LidarBevGeometricAugment",
            flip_prob=0.5,
            global_scale_noise=(0.95, 1.05),
            point_cloud_range=point_cloud_range,
            filter_outside=True,
        ),
        dict(type="BevBBoxRotation", rotation_3d_range=(-0.3925, 0.3925)),
    ],
)

data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=train_dataset,
    sampler=dict(type=torch.utils.data.DistributedSampler),
    batch_size=batch_size_per_gpu,
    shuffle=True,
    num_workers=4,
    pin_memory=False,
    collate_fn=collate_nuscenes_sequencev2,
)

val_dataset = dict(
    type="NuscenesBevSequenceDataset",
    data_path=val_data_path,
    map_size=None,
    map_path=None,
    with_bev_mask=False,
    with_lidar_bboxes=True,
    with_bev_bboxes=False,
    with_ego_bboxes=True,
    bev_range=point_cloud_range,
    with_lidar_occ=False,
    need_lidar=True,
    num_sweeps=1,
    load_dim=5,
    use_dim=[0, 1, 2, 3, 4],
    num_seq=1,
    transforms=[
        dict(type="MultiViewsImgResize", size=input_size),
        dict(
            type="MultiViewsImgTransformWrapper",
            transforms=[
                dict(type="PILToTensor"),
                dict(type="Pad", divisor=32),
                dict(type="BgrToYuv444", rgb_input=True),
                dict(type="Normalize", mean=128.0, std=128.0),
            ],
        ),
    ],
)

val_data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=val_dataset,
    sampler=dict(type=torch.utils.data.DistributedSampler),
    batch_size=1,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    collate_fn=collate_nuscenes_sequencev2,
)


def loss_collector(outputs: dict):
    losses = []
    for _, loss in outputs.items():
        losses.append(loss)
    return losses


batch_processor = dict(
    type="MultiBatchProcessor",
    need_grad_update=True,
    enable_amp=True,
    loss_collector=loss_collector,
)
val_batch_processor = dict(
    type="MultiBatchProcessor",
    need_grad_update=False,
    loss_collector=None,
)


def update_metric(metrics, batch, model_outs):
    metric_gt = {"meta": batch["seq_meta"][0]["meta"]}
    # BevFusion eval returns list[decoder_out]
    preds = model_outs[0] if isinstance(model_outs, (list, tuple)) else model_outs
    metrics[0].update(metric_gt, preds)


def update_loss(metrics, batch, model_outs):
    for metric in metrics:
        metric.update(model_outs)


val_metric_updater = dict(
    type="MetricUpdater",
    metric_update_func=update_metric,
    step_log_freq=10000,
    epoch_log_freq=1,
    log_prefix="Validation " + task_name,
)
loss_show_update = dict(
    type="MetricUpdater",
    metric_update_func=update_loss,
    step_log_freq=log_loss_show,
    epoch_log_freq=1,
    log_prefix="loss_" + task_name,
)

stat_callback = dict(
    type="StatsMonitor",
    log_freq=log_loss_show,
)

ckpt_callback = dict(
    type="Checkpoint",
    save_dir=ckpt_dir,
    name_prefix=training_step + "-",
    strict_match=True,
    save_interval=1,
    mode=None,
)

val_callback = dict(
    type="Validation",
    data_loader=val_data_loader,
    batch_processor=val_batch_processor,
    callbacks=[val_metric_updater],
    val_model=None,
    val_on_train_end=True,
    val_interval=2,
    log_interval=200,
)

trace_callback = dict(
    type="SaveTraced",
    save_dir=ckpt_dir,
    trace_inputs=deploy_inputs,
)

grad_callback = dict(
    type="GradScale",
    module_and_scale=[],
    clip_grad_norm=35,
    clip_norm_type=2,
)

val_nuscenes_metric = dict(
    type="NuscenesMetric",
    data_root=meta_rootdir,
    version="v1.0-trainval",
    use_lidar=True,
    classes=class_names,
    save_prefix="./metric_results/" + task_name,
    lidar_key="sensor2ego",
    trans_lidar_dim=True,
    trans_lidar_rot=False,
)


_float_converters = []
if os.path.isfile(lidar_pretrain_ckpt):
    _float_converters.append(
        dict(
            type="LoadCheckpoint",
            checkpoint_path=lidar_pretrain_ckpt,
            # state_dict_update_func=remap_lidar_centerpoint_state_dict,
            allow_miss=True,
            ignore_extra=True,
            verbose=True,
        )
    )

float_trainer = dict(
    type="distributed_data_parallel_trainer",
    model=model,
    data_loader=data_loader,
    # DETR/BEVFormer recipe (see bevfusion_centerpoint_* / bevformer_tiny):
    # - Cosine + warmup (not CenterPoint CyclicLr)
    # - lower LR on pretrained lidar_net, higher on query decoder
    # - 24 epochs for Hungarian matching to stabilize
    optimizer=dict(
        type=torch.optim.AdamW,
        betas=(0.95, 0.99),
        params={
            "lidar_net": dict(lr_mult=0.1),
            "bev_decoders": dict(lr_mult=3.0),
        },
        lr=2e-4,
        weight_decay=0.01,
    ),
    batch_processor=batch_processor,
    resume_optimizer=False,
    resume_epoch_or_step=False,
    num_epochs=128,
    start_epoch=24, 
    stop_by="epoch",
    device=None,
    callbacks=[
        stat_callback,
        loss_show_update,
        dict(
            type="CosineAnnealingLrUpdater",
            warmup_len=0,
            warmup_by="step",
            warmup_lr_ratio=1.0,
            step_log_interval=500,
            stop_lr=1e-6,
        ),
        grad_callback,
        val_callback,
        ckpt_callback,
    ],
    sync_bn=True,
    train_metrics=dict(
        type="LossShow",
    ),
    val_metrics=[val_nuscenes_metric],
)
if _float_converters:
    float_trainer["model_convert_pipeline"] = dict(
        type="ModelConvertPipeline",
        converters=_float_converters,
    )


calibration_data_loader = copy.deepcopy(data_loader)
calibration_data_loader.pop("sampler")
calibration_batch_processor = copy.deepcopy(val_batch_processor)

calibration_trainer = dict(
    type="Calibrator",
    model=model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode="fuse_bn",
        converters=[
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    ckpt_dir, "float-checkpoint-last.pth.tar"
                ),
                allow_miss=True,
                verbose=True,
            ),
            dict(type="Float2Calibration", convert_mode=convert_mode),
        ],
    ),
    data_loader=calibration_data_loader,
    batch_processor=calibration_batch_processor,
    num_steps=100,
    device=None,
    callbacks=[
        stat_callback,
        val_callback,
        ckpt_callback,
    ],
    val_metrics=[val_nuscenes_metric],
    log_interval=20,
)


qat_trainer = dict(
    type="distributed_data_parallel_trainer",
    model=model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode="fuse_bn",
        qconfig_params=dict(
            activation_qat_qkwargs=dict(
                averaging_constant=0,
            ),
            weight_qat_qkwargs=dict(
                averaging_constant=1,
            ),
        ),
        converters=[
            dict(type="Float2QAT", convert_mode=convert_mode),
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    ckpt_dir, "calibration-checkpoint-last.pth.tar"
                ),
            ),
        ],
    ),
    data_loader=data_loader,
    optimizer=dict(
        type=torch.optim.SGD,
        weight_decay=0.0,
        lr=2e-4,
        momentum=0.9,
    ),
    batch_processor=batch_processor,
    num_epochs=10,
    device=None,
    callbacks=[
        stat_callback,
        loss_show_update,
        dict(
            type="CyclicLrUpdater",
            target_ratio=(10, 1e-4),
            cyclic_times=1,
            step_ratio_up=0.4,
            step_log_interval=200,
        ),
        grad_callback,
        val_callback,
        ckpt_callback,
    ],
    train_metrics=dict(
        type="LossShow",
    ),
    val_metrics=[val_nuscenes_metric],
)

compile_dir = os.path.join(ckpt_dir, "compile")
compile_cfg = dict(
    march=march,
    name=task_name,
    hbm=os.path.join(compile_dir, "model.hbm"),
    layer_details=True,
    input_source=["ddr", "ddr"],
    opt="O2",
    output_layout="NHWC",
    enable_vpu=enable_vpu,
)


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
    metrics=[val_nuscenes_metric],
    callbacks=[
        val_metric_updater,
    ],
    log_interval=100,
)

calibration_predictor = dict(
    type="Predictor",
    model=model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode="fuse_bn",
        converters=[
            dict(type="Float2QAT", convert_mode=convert_mode),
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    ckpt_dir, "calibration-checkpoint-last.pth.tar"
                ),
            ),
        ],
    ),
    data_loader=[val_data_loader],
    batch_processor=val_batch_processor,
    device=None,
    metrics=[val_nuscenes_metric],
    callbacks=[
        val_metric_updater,
    ],
    log_interval=100,
)

qat_predictor = dict(
    type="Predictor",
    model=model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode="fuse_bn",
        converters=[
            dict(type="Float2QAT", convert_mode=convert_mode),
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    ckpt_dir, "qat-checkpoint-last.pth.tar"
                ),
            ),
            dict(type="SetQuantRoundingMode"),
        ],
    ),
    data_loader=[val_data_loader],
    batch_processor=val_batch_processor,
    device=None,
    metrics=[val_nuscenes_metric],
    callbacks=[
        val_metric_updater,
    ],
    log_interval=100,
)


def process_inputs(infer_inputs, transforms=None):
    points = np.load(os.path.join(infer_inputs, "points.npy")).reshape((-1, 5))
    points = torch.from_numpy(points)
    model_input = {
        "points": [points],
    }
    if transforms is not None:
        model_input = transforms(model_input)
    return model_input, points


def process_outputs(model_outs, viz_func, vis_inputs):
    preds = model_outs[0] if isinstance(model_outs, (list, tuple)) else model_outs
    if isinstance(preds, (list, tuple)):
        preds = preds[0]
    viz_func(vis_inputs, preds)
    return None


single_infer_dataset = copy.deepcopy(val_data_loader["dataset"])
single_infer_dataset["transforms"] = None


def inputs_save_func(data, save_path):
    if os.path.isdir(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path, exist_ok=True)
    points_path = os.path.join(save_path, "points.npy")
    # NuscenesBevSequenceDataset packs points under top-level or lidar
    if "points" in data:
        pts = data["points"]
    else:
        pts = data["lidar"]["points"]
    np.save(points_path, pts)


infer_cfg = dict(
    model=model,
    input_path=f"./demo/{task_name}",
    gen_inputs_cfg=dict(
        dataset=single_infer_dataset,
        sample_idx=[0],
        inputs_save_func=inputs_save_func,
    ),
    process_inputs=process_inputs,
    viz_func=partial(
        lidar_det_visualize, score_thresh=0.4, is_plot=True, reverse=True
    ),
    process_outputs=process_outputs,
)

onnx_cfg = dict(
    model=deploy_model,
    stage="qat",
    inputs=deploy_inputs,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode="fuse_bn",
        converters=[
            dict(type="Float2QAT"),
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    ckpt_dir, "qat-checkpoint-last.pth.tar"
                ),
            ),
        ],
    ),
)
