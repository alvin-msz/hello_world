python3 samples/ai_toolchain/horizon_model_train_sample/scripts/tools/train.py   --stage float   --config samples/ai_toolchain/horizon_model_train_sample/scripts/configs/lidar_bevfusion/bevfusion_pointpillar_henet_multisensor_multitask_nuscenes_argmax_cpu_ego.py   --device-ids 4,5,6,7

python3 samples/ai_toolchain/horizon_model_train_sample/scripts/tools/train.py   --stage calibration   --config samples/ai_toolchain/horizon_model_train_sample/scripts/configs/lidar_bevfusion/bevfusion_pointpillar_henet_multisensor_multitask_nuscenes_argmax_cpu_ego.py   --device-ids 4,5,6,7

python3 samples/ai_toolchain/horizon_model_train_sample/scripts/tools/train.py   --stage qat   --config samples/ai_toolchain/horizon_model_train_sample/scripts/configs/lidar_bevfusion/bevfusion_pointpillar_henet_multisensor_multitask_nuscenes_argmax_cpu_ego.py   --device-ids 4,5,6,7

python3 samples/ai_toolchain/horizon_model_train_sample/scripts/tools/export_hbir.py  --config samples/ai_toolchain/horizon_model_train_sample/scripts/configs/lidar_bevfusion/bevfusion_pointpillar_henet_multisensor_multitask_nuscenes_argmax_bpu_ego.py

python3 samples/ai_toolchain/horizon_model_train_sample/scripts/tools/compile_perf_hbir.py  --config samples/ai_toolchain/horizon_model_train_sample/scripts/configs/lidar_bevfusion/bevfusion_pointpillar_henet_multisensor_multitask_nuscenes_argmax_bpu_ego.py

python3 samples/ai_toolchain/horizon_model_train_sample/scripts/tools/infer_float_occ_one_frame.py -c samples/ai_toolchain/horizon_model_train_sample/scripts/configs/lidar_bevfusion/bevfusion_pointpillar_henet_multisensor_multitask_nuscenes_argmax_cpu_ego.py --ckpt  ./tmp_models/bevfusion_pointpillar_henet_multisensor_multitask_nuscenes_ego/float-checkpoint-last.pth.tar --infer-example-dir ./tmp_pretrained_models/bevfusion_pointpillar_henet_multisensor_multitask/ --point-coord-frame lidar  --output-dir ./tmp_eval/float_infer/bevfusion_ego/ --device cuda:4

python3 samples/ai_toolchain/horizon_model_train_sample/scripts/tools/quant_analysis.py -c samples/ai_toolchain/horizon_model_train_sample/scripts/configs/lidar_bevfusion/bevfusion_pointpillar_henet_multisensor_multitask_nuscenes_argmax_cpu_ego.py -ids 0