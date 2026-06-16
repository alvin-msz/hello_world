python3 samples/ai_toolchain/horizon_model_train_sample/scripts/tools/evaluate_nuscenes.py --config samples/ai_toolchain/horizon_model_train_sample/scripts/configs/custom/flashocc-r50-M0_bevfusionocc_horizon_2.py --occ-dir ./tmp_eval/custom_hbm/occ_preds/  --data-root data/nuscenes --output-dir ./tmp_eval/custom_hbm/occ_preds/ --frames-lst ./tmp_eval/custom_hbm/frames_0528.lst --ignore-index 17 --mask-type camera
`aidisdk` dependency is not available.
WARNING:__main__:--results-file and --data-root are required for mAP evaluation, skipping mAP evaluation.
INFO:__main__:Evaluating OCC mIOU from directory: ./tmp_eval/custom_hbm/occ_preds/
INFO:__main__:Found 6019 OCC bin files
INFO:__main__:Read 6019 entries from ./tmp_eval/custom_hbm/frames_0528.lst
INFO:__main__:Found 34149 GT files in data/nuscenes/occ3d/gts



INFO:__main__:Loaded 6019 OCC GT samples by frames.lst order (6019 with camera mask)
INFO:__main__:Using default occ3d-nuscenes class names (18 classes)
INFO:__main__:Using ignore_index=17
INFO:__main__:[Diag frame 0] pred.size=640000 dtype=int16, gt.size=640000, mask=size=640000 True=108100
INFO:__main__:[Diag frame 0] pred[:8]=[14, 14, 14, 14, 17, 17, 17, 17]  gt[:8]=[17, 17, 17, 17, 17, 17, 17, 17]
INFO:__main__:[Diag frame 0] after mask -> pred[:8]=[17, 17, 17, 4, 4, 4, 4, 4]  gt[:8]=[17, 17, 4, 4, 4, 4, 4, 4]
INFO:__main__:[Diag frame 0] gt class dist: {4: 779, 7: 85, 8: 18, 11: 7588, 12: 198, 13: 1577, 14: 2443, 15: 5801, 16: 1583, 17: 88028}
INFO:__main__:[Diag frame 0] pred class dist: {4: 812, 6: 11, 7: 25, 8: 1, 11: 8254, 12: 50, 13: 959, 14: 2497, 15: 5439, 16: 1373}
INFO:__main__:Loaded 6019 valid OCC frames
INFO:__main__:mIOU: 0.5117  (over 6019 frames)
INFO:__main__:  others: 0.1027
INFO:__main__:  barrier: 0.5931
INFO:__main__:  bicycle: 0.1995
INFO:__main__:  bus: 0.6670
INFO:__main__:  car: 0.6878
INFO:__main__:  construction_vehicle: 0.3347
INFO:__main__:  motorcycle: 0.3074
INFO:__main__:  pedestrian: 0.4785
INFO:__main__:  traffic_cone: 0.2775
INFO:__main__:  trailer: 0.4921
INFO:__main__:  truck: 0.5475
INFO:__main__:  driveable_surface: 0.8582
INFO:__main__:  other_flat: 0.4557
INFO:__main__:  sidewalk: 0.5743
INFO:__main__:  terrain: 0.6312
INFO:__main__:  manmade: 0.7445
INFO:__main__:  vegetation: 0.7474