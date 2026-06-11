python3 samples/ai_toolchain/horizon_model_train_sample/scripts/tools/evaluate_nuscenes.py --config samples/ai_toolchain/horizon_model_train_sample/scripts/configs/custom/flashocc-r50-M0_bevfusionocc_horizon_2.py --occ-dir ./tmp_eval/custom_hbm/occ_preds_4d/  --data-root data/nuscenes --output-dir ./tmp_eval/custom_hbm/occ_preds_4d/ --frames-lst ./tmp_eval/custom_hbm/frames.lst --ignore-index 17 --mask-type camera
`aidisdk` dependency is not available.
WARNING:__main__:--results-file and --data-root are required for mAP evaluation, skipping mAP evaluation.
INFO:__main__:Evaluating OCC mIOU from directory: ./tmp_eval/custom_hbm/occ_preds_4d/
INFO:__main__:Found 16 OCC bin files
INFO:__main__:Read 16 entries from ./tmp_eval/custom_hbm/frames.lst
INFO:__main__:Found 34149 GT files in data/nuscenes/occ3d/gts
WARNING:__main__:OCC GT not found for token prefix 
WARNING:__main__:OCC GT not found for token prefix 
WARNING:__main__:OCC GT not found for token prefix 
WARNING:__main__:OCC GT not found for token prefix 
WARNING:__main__:OCC GT not found for token prefix 
WARNING:__main__:Total 16 GT files not found (suppressed after 5)
INFO:__main__:Loaded 16 OCC GT samples by frames.lst order (0 with camera mask)
INFO:__main__:Using default occ3d-nuscenes class names (18 classes)
INFO:__main__:Using ignore_index=17
WARNING:__main__:Skipping ./tmp_eval/custom_hbm/occ_preds_4d/occ_rank0_000000.bin: GT is empty for index 0
WARNING:__main__:Skipping ./tmp_eval/custom_hbm/occ_preds_4d/occ_rank0_000001.bin: GT is empty for index 1
WARNING:__main__:Skipping ./tmp_eval/custom_hbm/occ_preds_4d/occ_rank0_000002.bin: GT is empty for index 2
WARNING:__main__:Skipping ./tmp_eval/custom_hbm/occ_preds_4d/occ_rank0_000003.bin: GT is empty for index 3
WARNING:__main__:Skipping ./tmp_eval/custom_hbm/occ_preds_4d/occ_rank0_000004.bin: GT is empty for index 4
WARNING:__main__:Skipping ./tmp_eval/custom_hbm/occ_preds_4d/occ_rank0_000005.bin: GT is empty for index 5
WARNING:__main__:Skipping ./tmp_eval/custom_hbm/occ_preds_4d/occ_rank0_000006.bin: GT is empty for index 6
WARNING:__main__:Skipping ./tmp_eval/custom_hbm/occ_preds_4d/occ_rank0_000007.bin: GT is empty for index 7
WARNING:__main__:Skipping ./tmp_eval/custom_hbm/occ_preds_4d/occ_rank0_000008.bin: GT is empty for index 8
WARNING:__main__:Skipping ./tmp_eval/custom_hbm/occ_preds_4d/occ_rank0_000009.bin: GT is empty for index 9
WARNING:__main__:Skipping ./tmp_eval/custom_hbm/occ_preds_4d/occ_rank0_000010.bin: GT is empty for index 10
WARNING:__main__:Skipping ./tmp_eval/custom_hbm/occ_preds_4d/occ_rank0_000011.bin: GT is empty for index 11
WARNING:__main__:Skipping ./tmp_eval/custom_hbm/occ_preds_4d/occ_rank0_000012.bin: GT is empty for index 12
WARNING:__main__:Skipping ./tmp_eval/custom_hbm/occ_preds_4d/occ_rank0_000013.bin: GT is empty for index 13
WARNING:__main__:Skipping ./tmp_eval/custom_hbm/occ_preds_4d/occ_rank0_000014.bin: GT is empty for index 14
WARNING:__main__:Skipping ./tmp_eval/custom_hbm/occ_preds_4d/occ_rank0_000015.bin: GT is empty for index 15
WARNING:__main__:No valid OCC samples found.
