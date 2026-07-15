2026-07-15 09:16:56,580 INFO [lr_updater.py:204] Node[0] Epoch[0] Step[0] GlobalStep[0] lr=0.000400
2026-07-15 09:21:38,767 INFO [monitor.py:131] Node[0] Epoch[0] Step[0-199] Cost Time: 282.187s Speed: 4.25 samples/sec Remaining Time: 11:00:36 Remaining step percent: 99.29%
2026-07-15 09:21:39,938 INFO [metric_updater.py:360] Node[0] Epoch[0] Step[199] GlobalStep[199] loss_bevfusion_centerpoint_pointpillar_henet_multisensor_multitask_nuscenes_opt: car_cls_loss[1.0837] car_reg_loss[0.8425] truck_cls_loss[3.4037] truck_reg_loss[0.9305] bus_cls_loss[9.7211] bus_reg_loss[1.0021] barrier_cls_loss[4.2308] barrier_reg_loss[0.8438] bicycle_cls_loss[19.7160] bicycle_reg_loss[0.8602] pedestrian_cls_loss[1.6458] pedestrian_reg_loss[0.9005] loss_occ[2.9068] 
2026-07-15 09:25:48,077 INFO [monitor.py:131] Node[0] Epoch[0] Step[200-399] Cost Time: 249.308s Speed: 4.81 samples/sec Remaining Time: 9:36:33 Remaining step percent: 98.58%
2026-07-15 09:25:49,484 INFO [metric_updater.py:360] Node[0] Epoch[0] Step[399] GlobalStep[399] loss_bevfusion_centerpoint_pointpillar_henet_multisensor_multitask_nuscenes_opt: car_cls_loss[0.7912] car_reg_loss[0.7341] truck_cls_loss[2.1164] truck_reg_loss[0.8205] bus_cls_loss[5.2479] bus_reg_loss[0.8685] barrier_cls_loss[2.4213] barrier_reg_loss[0.7539] bicycle_cls_loss[10.3757] bicycle_reg_loss[0.8012] pedestrian_cls_loss[1.0666] pedestrian_reg_loss[0.8221] loss_occ[2.3895] 
2026-07-15 09:27:53,877 INFO [lr_updater.py:204] Node[0] Epoch[0] Step[499] GlobalStep[499] lr=0.000399
2026-07-15 09:30:07,702 INFO [monitor.py:131] Node[0] Epoch[0] Step[400-599] Cost Time: 259.622s Speed: 4.62 samples/sec Remaining Time: 9:56:05 Remaining step percent: 97.87%
2026-07-15 09:30:08,752 INFO [metric_updater.py:360] Node[0] Epoch[0] Step[599] GlobalStep[599] loss_bevfusion_centerpoint_pointpillar_henet_multisensor_multitask_nuscenes_opt: car_cls_loss[0.6910] car_reg_loss[0.6723] truck_cls_loss[1.6642] truck_reg_loss[0.7516] bus_cls_loss[3.7186] bus_reg_loss[0.8025] barrier_cls_loss[1.8119] barrier_reg_loss[0.7081] bicycle_cls_loss[7.1937] bicycle_reg_loss[0.7557] pedestrian_cls_loss[0.8657] pedestrian_reg_loss[0.7865] loss_occ[2.2154] 
2026-07-15 09:34:34,408 INFO [monitor.py:131] Node[0] Epoch[0] Step[600-799] Cost Time: 266.704s Speed: 4.50 samples/sec Remaining Time: 10:07:54 Remaining step percent: 97.16%
2026-07-15 09:34:35,751 INFO [metric_updater.py:360] Node[0] Epoch[0] Step[799] GlobalStep[799] loss_bevfusion_centerpoint_pointpillar_henet_multisensor_multitask_nuscenes_opt: car_cls_loss[0.6401] car_reg_loss[0.6296] truck_cls_loss[1.4294] truck_reg_loss[0.7078] bus_cls_loss[2.9534] bus_reg_loss[0.7605] barrier_cls_loss[1.5147] barrier_reg_loss[0.6818] bicycle_cls_loss[5.6117] bicycle_reg_loss[0.7340] pedestrian_cls_loss[0.7633] pedestrian_reg_loss[0.7623] loss_occ[2.1122] 
2026-07-15 09:39:01,612 INFO [monitor.py:131] Node[0] Epoch[0] Step[800-999] Cost Time: 267.203s Speed: 4.49 samples/sec Remaining Time: 10:04:35 Remaining step percent: 96.45%
2026-07-15 09:39:02,875 INFO [metric_updater.py:360] Node[0] Epoch[0] Step[999] GlobalStep[999] loss_bevfusion_centerpoint_pointpillar_henet_multisensor_multitask_nuscenes_opt: car_cls_loss[0.6052] car_reg_loss[0.5978] truck_cls_loss[1.2904] truck_reg_loss[0.6755] bus_cls_loss[2.4799] bus_reg_loss[0.7283] barrier_cls_loss[1.3135] barrier_reg_loss[0.6613] bicycle_cls_loss[4.6463] bicycle_reg_loss[0.7075] pedestrian_cls_loss[0.7002] pedestrian_reg_loss[0.7442] loss_occ[2.0384] 
2026-07-15 09:42:56,966 INFO [monitor.py:146] Node[0] Epoch[0] End   ==================================================
2026-07-15 09:42:56,968 INFO [monitor.py:149] Node[0] Epoch[0] Cost Time: 1560.388s
2026-07-15 09:42:56,968 INFO [metric_updater.py:360] Node[0] Epoch[0] loss_bevfusion_centerpoint_pointpillar_henet_multisensor_multitask_nuscenes_opt: car_cls_loss[0.5867] car_reg_loss[0.5785] truck_cls_loss[1.2085] truck_reg_loss[0.6538] bus_cls_loss[2.2070] bus_reg_loss[0.7013] barrier_cls_loss[1.2021] barrier_reg_loss[0.6529] bicycle_cls_loss[4.0918] bicycle_reg_loss[0.6914] pedestrian_cls_loss[0.6608] pedestrian_reg_loss[0.7308] loss_occ[1.9962] 
2026-07-15 09:42:56,968 INFO [validation.py:156] Node[0] [32mUse train `model` as val model.[0m
2026-07-15 09:42:57,150 INFO [loop_base.py:482] Node[0] [32mStart Predictor loop from epoch 0, num_epochs=1[0m
2026-07-15 09:42:57,153 INFO [fake_quantize.py:253] Node[0] Set fake quantize state to FakeQuantState.VALIDATION
2026-07-15 09:43:06,387 INFO [loop_base.py:553] Node[0] 0 / 1505
2026-07-15 09:43:35,999 INFO [loop_base.py:553] Node[0] 200 / 1505
2026-07-15 09:44:02,180 INFO [loop_base.py:553] Node[0] 400 / 1505
2026-07-15 09:44:28,526 INFO [loop_base.py:553] Node[0] 600 / 1505
2026-07-15 09:44:55,848 INFO [loop_base.py:553] Node[0] 800 / 1505
2026-07-15 09:45:22,611 INFO [loop_base.py:553] Node[0] 1000 / 1505
2026-07-15 09:45:48,273 INFO [loop_base.py:553] Node[0] 1200 / 1505
2026-07-15 09:46:15,702 INFO [loop_base.py:553] Node[0] 1400 / 1505
2026-07-15 09:46:31,304 INFO [nuscenes_metric.py:322] Node[0] The length of self.nusc_annos is: 1505
2026-07-15 09:46:34,663 INFO [nuscenes_metric.py:346] Node[0] Results writes to ./metric_results/bevfusion_centerpoint_pointpillar_henet_multisensor_multitask_nuscenes_opt/results_nusc.json
2026-07-15 09:48:58,507 INFO [nuscenes_metric.py:388] Node[0] NDS: 0.5038, mAP:0.4827
car_AP: [0.5]:0.5923  [1.0]:0.8143  [2.0]:0.8736  [4.0]:0.8941 
truck_AP: [0.5]:0.2145  [1.0]:0.4729  [2.0]:0.5966  [4.0]:0.6392 
trailer_AP: [0.5]:0.0404  [1.0]:0.2506  [2.0]:0.4111  [4.0]:0.4795 
bus_AP: [0.5]:0.2319  [1.0]:0.6056  [2.0]:0.7515  [4.0]:0.7952 
construction_vehicle_AP: [0.5]:0.0043  [1.0]:0.1230  [2.0]:0.2894  [4.0]:0.3431 
bicycle_AP: [0.5]:0.2196  [1.0]:0.3977  [2.0]:0.4230  [4.0]:0.4413 
motorcycle_AP: [0.5]:0.3174  [1.0]:0.6149  [2.0]:0.6475  [4.0]:0.6630 
pedestrian_AP: [0.5]:0.3617  [1.0]:0.5441  [2.0]:0.5718  [4.0]:0.6212 
traffic_cone_AP: [0.5]:0.3518  [1.0]:0.5492  [2.0]:0.5988  [4.0]:0.6585 
barrier_AP: [0.5]:0.1363  [1.0]:0.5101  [2.0]:0.6062  [4.0]:0.6509 

2026-07-15 09:48:58,781 WARNING [metric.py:200] Node[0] <class 'hat.metrics.nuscenes_metric.NuscenesMetric'> not ready for distributed environment, should not be used together with DistributedSampler.Might be slow in validation due to resource competition
2026-07-15 09:48:58,816 INFO [mean_iou.py:170] Node[0] ~~~~ MeanIOU Summary metrics ~~~~
Summary:
Scope                 mIoU       mAcc       aAcc
global               44.13      52.35      71.55
Per Class Results:
Class                  IoU        Acc
others                1.28       1.30
barrier              48.99      65.65
bicycle              27.20      32.01
bus                  61.44      78.40
car                  66.41      75.27
construction_vehicle      30.80      36.71
motorcycle           35.37      38.31
pedestrian           51.72      60.17
traffic_cone         22.11      25.98
trailer              38.85      51.71
truck                49.21      58.29
driveable_surface      76.26      85.14
other_flat           23.74      26.35
sidewalk             38.67      46.93
terrain              48.85      63.79
manmade              63.41      73.35
vegetation           65.98      70.53

2026-07-15 09:48:58,834 INFO [metric_updater.py:360] Node[0] Epoch[0] Validation bevfusion_centerpoint_pointpillar_henet_multisensor_multitask_nuscenes_opt: NDS[0.5038] MeanIOU[tensor(0.4413, device='cuda:0')] 
2026-07-15 09:49:06,891 INFO [checkpoint.py:233] Node[0] Save model checkpoint: ./tmp_models/bevfusion_centerpoint_pointpillar_henet_multisensor_multitask_nuscenes_opt/float-checkpoint-epoch-0000-dbaf3229.pth.tar
2026-07-15 09:49:07,512 INFO [checkpoint.py:270] Node[0] Save last model checkpoint: ./tmp_models/bevfusion_centerpoint_pointpillar_henet_multisensor_multitask_nuscenes_opt/float-checkpoint-last-2ca36d2b.pth.tar
2026-07-15 09:49:07,520 INFO [monitor.py:143] Node[0] Epoch[1] Begin ==================================================
2026-07-15 09:49:07,520 INFO [lr_updater.py:204] Node[0] Epoch[1] Step[0] GlobalStep[1173] lr=0.000398
2026-07-15 09:53:34,762 INFO [monitor.py:131] Node[0] Epoch[1] Step[0-199] Cost Time: 267.239s Speed: 4.49 samples/sec Remaining Time: 9:59:21 Remaining step percent: 95.12%
2026-07-15 09:53:35,871 INFO [metric_updater.py:360] Node[0] Epoch[1] Step[199] GlobalStep[1372] loss_bevfusion_centerpoint_pointpillar_henet_multisensor_multitask_nuscenes_opt: car_cls_loss[0.4542] car_reg_loss[0.4559] truck_cls_loss[0.6479] truck_reg_loss[0.5208] bus_cls_loss[0.5338] bus_reg_loss[0.5225] barrier_cls_loss[0.5794] barrier_reg_loss[0.5498] bicycle_cls_loss[0.7155] bicycle_reg_loss[0.5841] pedestrian_cls_loss[0.4275] pedestrian_reg_loss[0.6481] loss_occ[1.7060] 
2026-07-15 09:57:40,365 INFO [monitor.py:131] Node[0] Epoch[1] Step[200-399] Cost Time: 245.601s Speed: 4.89 samples/sec Remaining Time: 9:03:59 Remaining step percent: 94.41%
2026-07-15 09:57:41,554 INFO [metric_updater.py:360] Node[0] Epoch[1] Step[399] GlobalStep[1572] loss_bevfusion_centerpoint_pointpillar_henet_multisensor_multitask_nuscenes_opt: car_cls_loss[0.4534] car_reg_loss[0.4488] truck_cls_loss[0.6582] truck_reg_loss[0.5069] bus_cls_loss[0.5406] bus_reg_loss[0.5235] barrier_cls_loss[0.5523] barrier_reg_loss[0.5463] bicycle_cls_loss[0.7087] bicycle_reg_loss[0.5705] pedestrian_cls_loss[0.4266] pedestrian_reg_loss[0.6433] loss_occ[1.7050] 
2026-07-15 10:01:47,560 INFO [monitor.py:131] Node[0] Epoch[1] Step[400-599] Cost Time: 247.194s Speed: 4.85 samples/sec Remaining Time: 9:03:23 Remaining step percent: 93.70%
2026-07-15 10:01:48,835 INFO [metric_updater.py:360] Node[0] Epoch[1] Step[599] GlobalStep[1772] loss_bevfusion_centerpoint_pointpillar_henet_multisensor_multitask_nuscenes_opt: car_cls_loss[0.4549] car_reg_loss[0.4444] truck_cls_loss[0.6648] truck_reg_loss[0.4994] bus_cls_loss[0.5132] bus_reg_loss[0.5087] barrier_cls_loss[0.5425] barrier_reg_loss[0.5423] bicycle_cls_loss[0.7012] bicycle_reg_loss[0.5698] pedestrian_cls_loss[0.4228] pedestrian_reg_loss[0.6388] loss_occ[1.7057] 
2026-07-15 10:05:57,816 INFO [monitor.py:131] Node[0] Epoch[1] Step[600-799] Cost Time: 250.254s Speed: 4.80 samples/sec Remaining Time: 9:05:56 Remaining step percent: 92.99%
2026-07-15 10:05:59,037 INFO [metric_updater.py:360] Node[0] Epoch[1] Step[799] GlobalStep[1972] loss_bevfusion_centerpoint_pointpillar_henet_multisensor_multitask_nuscenes_opt: car_cls_loss[0.4547] car_reg_loss[0.4409] truck_cls_loss[0.6609] truck_reg_loss[0.4944] bus_cls_loss[0.5071] bus_reg_loss[0.5021] barrier_cls_loss[0.5499] barrier_reg_loss[0.5464] bicycle_cls_loss[0.6947] bicycle_reg_loss[0.5604] pedestrian_cls_loss[0.4223] pedestrian_reg_loss[0.6369] loss_occ[1.6901] 
2026-07-15 10:10:20,736 INFO [monitor.py:131] Node[0] Epoch[1] Step[800-999] Cost Time: 262.919s Speed: 4.56 samples/sec Remaining Time: 9:29:11 Remaining step percent: 92.28%
2026-07-15 10:10:22,014 INFO [metric_updater.py:360] Node[0] Epoch[1] Step[999] GlobalStep[2172] loss_bevfusion_centerpoint_pointpillar_henet_multisensor_multitask_nuscenes_opt: car_cls_loss[0.4502] car_reg_loss[0.4371] truck_cls_loss[0.6576] truck_reg_loss[0.4902] bus_cls_loss[0.4993] bus_reg_loss[0.5021] barrier_cls_loss[0.5355] barrier_reg_loss[0.5464] bicycle_cls_loss[0.6794] bicycle_reg_loss[0.5536] pedestrian_cls_loss[0.4175] pedestrian_reg_loss[0.6333] loss_occ[1.6845] 
2026-07-15 10:14:06,401 INFO [monitor.py:146] Node[0] Epoch[1] End   ==================================================
2026-07-15 10:14:06,403 INFO [monitor.py:149] Node[0] Epoch[1] Cost Time: 1498.883s
2026-07-15 10:14:06,404 INFO [metric_updater.py:360] Node[0] Epoch[1] loss_bevfusion_centerpoint_pointpillar_henet_multisensor_multitask_nuscenes_opt: car_cls_loss[0.4484] car_reg_loss[0.4351] truck_cls_loss[0.6549] truck_reg_loss[0.4878] bus_cls_loss[0.5063] bus_reg_loss[0.5002] barrier_cls_loss[0.5268] barrier_reg_loss[0.5420] bicycle_cls_loss[0.6698] bicycle_reg_loss[0.5459] pedestrian_cls_loss[0.4168] pedestrian_reg_loss[0.6315] loss_occ[1.6788] 
2026-07-15 10:14:06,404 INFO [validation.py:156] Node[0] [32mUse train `model` as val model.[0m
2026-07-15 10:14:06,626 INFO [loop_base.py:482] Node[0] [32mStart Predictor loop from epoch 1, num_epochs=1[0m
2026-07-15 10:14:06,628 INFO [fake_quantize.py:253] Node[0] Set fake quantize state to FakeQuantState.VALIDATION
2026-07-15 10:14:15,817 INFO [loop_base.py:553] Node[0] 0 / 1505
2026-07-15 10:14:42,321 INFO [loop_base.py:553] Node[0] 200 / 1505
2026-07-15 10:15:09,118 INFO [loop_base.py:553] Node[0] 400 / 1505
2026-07-15 10:15:49,547 INFO [loop_base.py:553] Node[0] 600 / 1505
2026-07-15 10:16:20,068 INFO [loop_base.py:553] Node[0] 800 / 1505
2026-07-15 10:16:55,842 INFO [loop_base.py:553] Node[0] 1000 / 1505
2026-07-15 10:17:22,892 INFO [loop_base.py:553] Node[0] 1200 / 1505
2026-07-15 10:17:49,238 INFO [loop_base.py:553] Node[0] 1400 / 1505
2026-07-15 10:18:09,928 INFO [nuscenes_metric.py:322] Node[0] The length of self.nusc_annos is: 1505
2026-07-15 10:18:12,672 INFO [nuscenes_metric.py:346] Node[0] Results writes to ./metric_results/bevfusion_centerpoint_pointpillar_henet_multisensor_multitask_nuscenes_opt/results_nusc.json
2026-07-15 10:20:26,612 INFO [nuscenes_metric.py:388] Node[0] NDS: 0.5525, mAP:0.5008
car_AP: [0.5]:0.6423  [1.0]:0.8291  [2.0]:0.8791  [4.0]:0.8984 
truck_AP: [0.5]:0.2768  [1.0]:0.5070  [2.0]:0.6263  [4.0]:0.6601 
trailer_AP: [0.5]:0.0411  [1.0]:0.2671  [2.0]:0.4257  [4.0]:0.5338 
bus_AP: [0.5]:0.3318  [1.0]:0.6542  [2.0]:0.7958  [4.0]:0.8363 
construction_vehicle_AP: [0.5]:0.0013  [1.0]:0.1054  [2.0]:0.2870  [4.0]:0.4034 
bicycle_AP: [0.5]:0.2317  [1.0]:0.3902  [2.0]:0.4111  [4.0]:0.4330 
motorcycle_AP: [0.5]:0.3451  [1.0]:0.6187  [2.0]:0.6589  [4.0]:0.6781 
pedestrian_AP: [0.5]:0.3711  [1.0]:0.5632  [2.0]:0.5921  [4.0]:0.6395 
traffic_cone_AP: [0.5]:0.3962  [1.0]:0.5699  [2.0]:0.6179  [4.0]:0.6688 
barrier_AP: [0.5]:0.1777  [1.0]:0.4864  [2.0]:0.5734  [4.0]:0.6066 

2026-07-15 10:20:26,815 WARNING [metric.py:200] Node[0] <class 'hat.metrics.nuscenes_metric.NuscenesMetric'> not ready for distributed environment, should not be used together with DistributedSampler.Might be slow in validation due to resource competition
2026-07-15 10:20:26,832 INFO [mean_iou.py:170] Node[0] ~~~~ MeanIOU Summary metrics ~~~~
Summary:
Scope                 mIoU       mAcc       aAcc
global               43.38      50.48      70.05
Per Class Results:
Class                  IoU        Acc
others                3.18       3.35
barrier              47.08      59.08
bicycle              19.00      20.46
bus                  61.78      65.99
car                  65.76      74.39
construction_vehicle      30.56      37.17
motorcycle           37.84      42.68
pedestrian           47.73      54.93
traffic_cone         22.67      26.59
trailer              42.06      52.30
truck                51.44      62.89
driveable_surface      75.89      82.15
other_flat           24.69      27.95
sidewalk             36.48      42.18
terrain              44.88      64.83
manmade              59.19      63.29
vegetation           67.27      77.93

2026-07-15 10:20:26,833 INFO [metric_updater.py:360] Node[0] Epoch[1] Validation bevfusion_centerpoint_pointpillar_henet_multisensor_multitask_nuscenes_opt: NDS[0.5525] MeanIOU[tensor(0.4338, device='cuda:0')] 
2026-07-15 10:20:34,079 INFO [checkpoint.py:233] Node[0] Save model checkpoint: ./tmp_models/bevfusion_centerpoint_pointpillar_henet_multisensor_multitask_nuscenes_opt/float-checkpoint-epoch-0001-e3f5c722.pth.tar
2026-07-15 10:20:35,076 INFO [checkpoint.py:270] Node[0] Save last model checkpoint: ./tmp_models/bevfusion_centerpoint_pointpillar_henet_multisensor_multitask_nuscenes_opt/float-checkpoint-last-849d1f5b.pth.tar
2026-07-15 10:20:35,084 INFO [monitor.py:143] Node[0] Epoch[2] Begin ==================================================
2026-07-15 10:20:35,084 INFO [lr_updater.py:204] Node[0] Epoch[2] Step[0] GlobalStep[2346] lr=0.000393
2026-07-15 10:25:16,136 INFO [monitor.py:131] Node[0] Epoch[2] Step[0-199] Cost Time: 281.049s Speed: 4.27 samples/sec Remaining Time: 10:02:43 Remaining step percent: 90.96%
2026-07-15 10:25:17,265 INFO [metric_updater.py:360] Node[0] Epoch[2] Step[199] GlobalStep[2545] loss_bevfusion_centerpoint_pointpillar_henet_multisensor_multitask_nuscenes_opt: car_cls_loss[0.4358] car_reg_loss[0.4203] truck_cls_loss[0.6009] truck_reg_loss[0.4598] bus_cls_loss[0.4839] bus_reg_loss[0.4629] barrier_cls_loss[0.4902] barrier_reg_loss[0.5172] bicycle_cls_loss[0.5969] bicycle_reg_loss[0.5119] pedestrian_cls_loss[0.3910] pedestrian_reg_loss[0.6173] loss_occ[1.6213] 
2026-07-15 10:29:25,816 INFO [monitor.py:131] Node[0] Epoch[2] Step[200-399] Cost Time: 249.679s Speed: 4.81 samples/sec Remaining Time: 8:48:36 Remaining step percent: 90.25%
2026-07-15 10:29:26,977 INFO [metric_updater.py:360] Node[0] Epoch[2] Step[399] GlobalStep[2745] loss_bevfusion_centerpoint_pointpillar_henet_multisensor_multitask_nuscenes_opt: car_cls_loss[0.4383] car_reg_loss[0.4180] truck_cls_loss[0.5901] truck_reg_loss[0.4578] bus_cls_loss[0.4790] bus_reg_loss[0.4680] barrier_cls_loss[0.4878] barrier_reg_loss[0.5292] bicycle_cls_loss[0.5763] bicycle_reg_loss[0.5093] pedestrian_cls_loss[0.3942] pedestrian_reg_loss[0.6159] loss_occ[1.6142] 
2026-07-15 10:33:30,037 INFO [monitor.py:131] Node[0] Epoch[2] Step[400-599] Cost Time: 244.220s Speed: 4.91 samples/sec Remaining Time: 8:32:59 Remaining step percent: 89.54%
2026-07-15 10:33:31,220 INFO [metric_updater.py:360] Node[0] Epoch[2] Step[599] GlobalStep[2945] loss_bevfusion_centerpoint_pointpillar_henet_multisensor_multitask_nuscenes_opt: car_cls_loss[0.4380] car_reg_loss[0.4157] truck_cls_loss[0.5927] truck_reg_loss[0.4540] bus_cls_loss[0.4573] bus_reg_loss[0.4571] barrier_cls_loss[0.4860] barrier_reg_loss[0.5257] bicycle_cls_loss[0.5594] bicycle_reg_loss[0.4936] pedestrian_cls_loss[0.3975] pedestrian_reg_loss[0.6161] loss_occ[1.6175] 
2026-07-15 10:37:49,312 INFO [monitor.py:131] Node[0] Epoch[2] Step[600-799] Cost Time: 259.273s Speed: 4.63 samples/sec Remaining Time: 9:00:16 Remaining step percent: 88.82%
2026-07-15 10:37:50,562 INFO [metric_updater.py:360] Node[0] Epoch[2] Step[799] GlobalStep[3145] loss_bevfusion_centerpoint_pointpillar_henet_multisensor_multitask_nuscenes_opt: car_cls_loss[0.4361] car_reg_loss[0.4134] truck_cls_loss[0.5877] truck_reg_loss[0.4509] bus_cls_loss[0.4609] bus_reg_loss[0.4581] barrier_cls_loss[0.4780] barrier_reg_loss[0.5198] bicycle_cls_loss[0.5610] bicycle_reg_loss[0.4913] pedestrian_cls_loss[0.3999] pedestrian_reg_loss[0.6152] loss_occ[1.6138] 
2026-07-15 10:42:06,032 INFO [monitor.py:131] Node[0] Epoch[2] Step[800-999] Cost Time: 256.718s Speed: 4.67 samples/sec Remaining Time: 8:50:40 Remaining step percent: 88.11%
2026-07-15 10:42:07,199 INFO [metric_updater.py:360] Node[0] Epoch[2] Step[999] GlobalStep[3345] loss_bevfusion_centerpoint_pointpillar_henet_multisensor_multitask_nuscenes_opt: car_cls_loss[0.4343] car_reg_loss[0.4124] truck_cls_loss[0.5884] truck_reg_loss[0.4495] bus_cls_loss[0.4599] bus_reg_loss[0.4551] barrier_cls_loss[0.4801] barrier_reg_loss[0.5155] bicycle_cls_loss[0.5594] bicycle_reg_loss[0.4914] pedestrian_cls_loss[0.3995] pedestrian_reg_loss[0.6134] loss_occ[1.6151] 
2026-07-15 10:45:49,530 INFO [monitor.py:146] Node[0] Epoch[2] End   ==================================================
2026-07-15 10:45:49,531 INFO [monitor.py:149] Node[0] Epoch[2] Cost Time: 1514.447s
2026-07-15 10:45:49,532 INFO [metric_updater.py:360] Node[0] Epoch[2] loss_bevfusion_centerpoint_pointpillar_henet_multisensor_multitask_nuscenes_opt: car_cls_loss[0.4334] car_reg_loss[0.4115] truck_cls_loss[0.5862] truck_reg_loss[0.4475] bus_cls_loss[0.4563] bus_reg_loss[0.4548] barrier_cls_loss[0.4768] barrier_reg_loss[0.5136] bicycle_cls_loss[0.5709] bicycle_reg_loss[0.4894] pedestrian_cls_loss[0.3974] pedestrian_reg_loss[0.6112] loss_occ[1.6153] 
2026-07-15 10:45:49,532 INFO [validation.py:156] Node[0] [32mUse train `model` as val model.[0m
2026-07-15 10:45:49,703 INFO [loop_base.py:482] Node[0] [32mStart Predictor loop from epoch 2, num_epochs=1[0m
2026-07-15 10:45:49,706 INFO [fake_quantize.py:253] Node[0] Set fake quantize state to FakeQuantState.VALIDATION
2026-07-15 10:45:57,370 INFO [loop_base.py:553] Node[0] 0 / 1505
2026-07-15 10:46:24,678 INFO [loop_base.py:553] Node[0] 200 / 1505
2026-07-15 10:47:02,908 INFO [loop_base.py:553] Node[0] 400 / 1505
2026-07-15 10:47:35,702 INFO [loop_base.py:553] Node[0] 600 / 1505
2026-07-15 10:48:17,233 INFO [loop_base.py:553] Node[0] 800 / 1505
2026-07-15 10:48:56,662 INFO [loop_base.py:553] Node[0] 1000 / 1505
2026-07-15 10:49:23,939 INFO [loop_base.py:553] Node[0] 1200 / 1505
2026-07-15 10:49:58,232 INFO [loop_base.py:553] Node[0] 1400 / 1505
2026-07-15 10:50:13,564 INFO [nuscenes_metric.py:322] Node[0] The length of self.nusc_annos is: 1505
2026-07-15 10:50:16,322 INFO [nuscenes_metric.py:346] Node[0] Results writes to ./metric_results/bevfusion_centerpoint_pointpillar_henet_multisensor_multitask_nuscenes_opt/results_nusc.json
2026-07-15 10:52:24,857 INFO [nuscenes_metric.py:388] Node[0] NDS: 0.5647, mAP:0.5010
car_AP: [0.5]:0.6355  [1.0]:0.8162  [2.0]:0.8674  [4.0]:0.8875 
truck_AP: [0.5]:0.2917  [1.0]:0.5064  [2.0]:0.6044  [4.0]:0.6375 
trailer_AP: [0.5]:0.0485  [1.0]:0.2728  [2.0]:0.4173  [4.0]:0.5396 
bus_AP: [0.5]:0.3412  [1.0]:0.6721  [2.0]:0.7949  [4.0]:0.8300 
construction_vehicle_AP: [0.5]:0.0027  [1.0]:0.1096  [2.0]:0.2702  [4.0]:0.3725 
bicycle_AP: [0.5]:0.2619  [1.0]:0.3792  [2.0]:0.3919  [4.0]:0.4066 
motorcycle_AP: [0.5]:0.3719  [1.0]:0.5995  [2.0]:0.6299  [4.0]:0.6423 
pedestrian_AP: [0.5]:0.3762  [1.0]:0.5507  [2.0]:0.5818  [4.0]:0.6227 
traffic_cone_AP: [0.5]:0.4215  [1.0]:0.5744  [2.0]:0.6176  [4.0]:0.6765 
barrier_AP: [0.5]:0.2341  [1.0]:0.5294  [2.0]:0.6120  [4.0]:0.6424 

2026-07-15 10:52:25,080 WARNING [metric.py:200] Node[0] <class 'hat.metrics.nuscenes_metric.NuscenesMetric'> not ready for distributed environment, should not be used together with DistributedSampler.Might be slow in validation due to resource competition
2026-07-15 10:52:25,100 INFO [mean_iou.py:170] Node[0] ~~~~ MeanIOU Summary metrics ~~~~
Summary:
Scope                 mIoU       mAcc       aAcc
global               46.15      56.06      73.39
Per Class Results:
Class                  IoU        Acc
others                4.26       4.71
barrier              51.24      64.95
bicycle              26.20      30.82
bus                  68.80      76.67
car                  68.88      75.93
construction_vehicle      32.72      42.32
motorcycle           34.56      38.10
pedestrian           52.85      59.93
traffic_cone         27.42      32.87
trailer              36.96      73.54
truck                50.49      60.34
driveable_surface      77.07      83.39
other_flat           30.76      37.37
sidewalk             42.98      55.47
terrain              47.50      70.04
manmade              62.94      67.80
vegetation           68.97      78.68

2026-07-15 10:52:25,102 INFO [metric_updater.py:360] Node[0] Epoch[2] Validation bevfusion_centerpoint_pointpillar_henet_multisensor_multitask_nuscenes_opt: NDS[0.5647] MeanIOU[tensor(0.4615, device='cuda:0')] 
2026-07-15 10:52:32,460 INFO [checkpoint.py:233] Node[0] Save model checkpoint: ./tmp_models/bevfusion_centerpoint_pointpillar_henet_multisensor_multitask_nuscenes_opt/float-checkpoint-epoch-0002-d455e029.pth.tar
2026-07-15 10:52:33,352 INFO [checkpoint.py:270] Node[0] Save last model checkpoint: ./tmp_models/bevfusion_centerpoint_pointpillar_henet_multisensor_multitask_nuscenes_opt/float-checkpoint-last-e2b9419b.pth.tar
2026-07-15 10:52:33,360 INFO [monitor.py:143] Node[0] Epoch[3] Begin ==================================================
2026-07-15 10:52:33,361 INFO [lr_updater.py:204] Node[0] Epoch[3] Step[0] GlobalStep[3519] lr=0.000385
2026-07-15 10:57:18,281 INFO [monitor.py:131] Node[0] Epoch[3] Step[0-199] Cost Time: 284.918s Speed: 4.21 samples/sec Remaining Time: 9:43:01 Remaining step percent: 86.79%
2026-07-15 10:57:19,615 INFO [metric_updater.py:360] Node[0] Epoch[3] Step[199] GlobalStep[3718] loss_bevfusion_centerpoint_pointpillar_henet_multisensor_multitask_nuscenes_opt: car_cls_loss[0.4384] car_reg_loss[0.4033] truck_cls_loss[0.5391] truck_reg_loss[0.4226] bus_cls_loss[0.3868] bus_reg_loss[0.4371] barrier_cls_loss[0.4353] barrier_reg_loss[0.5155] bicycle_cls_loss[0.4478] bicycle_reg_loss[0.4696] pedestrian_cls_loss[0.3959] pedestrian_reg_loss[0.5904] loss_occ[1.5780] 
2026-07-15 11:01:35,220 INFO [monitor.py:131] Node[0] Epoch[3] Step[200-399] Cost Time: 256.937s Speed: 4.67 samples/sec Remaining Time: 8:38:51 Remaining step percent: 86.08%
2026-07-15 11:01:36,612 INFO [metric_updater.py:360] Node[0] Epoch[3] Step[399] GlobalStep[3918] loss_bevfusion_centerpoint_pointpillar_henet_multisensor_multitask_nuscenes_opt: car_cls_loss[0.4322] car_reg_loss[0.4021] truck_cls_loss[0.5380] truck_reg_loss[0.4286] bus_cls_loss[0.3865] bus_reg_loss[0.4339] barrier_cls_loss[0.4378] barrier_reg_loss[0.5087] bicycle_cls_loss[0.4217] bicycle_reg_loss[0.4545] pedestrian_cls_loss[0.3856] pedestrian_reg_loss[0.5925] loss_occ[1.5816] 
2026-07-15 11:05:56,517 INFO [monitor.py:131] Node[0] Epoch[3] Step[400-599] Cost Time: 261.295s Speed: 4.59 samples/sec Remaining Time: 8:43:18 Remaining step percent: 85.37%
2026-07-15 11:05:57,756 INFO [metric_updater.py:360] Node[0] Epoch[3] Step[599] GlobalStep[4118] loss_bevfusion_centerpoint_pointpillar_henet_multisensor_multitask_nuscenes_opt: car_cls_loss[0.4311] car_reg_loss[0.4016] truck_cls_loss[0.5369] truck_reg_loss[0.4262] bus_cls_loss[0.3983] bus_reg_loss[0.4273] barrier_cls_loss[0.4383] barrier_reg_loss[0.4970] bicycle_cls_loss[0.4389] bicycle_reg_loss[0.4538] pedestrian_cls_loss[0.3866] pedestrian_reg_loss[0.5952] loss_occ[1.5816] 
2026-07-15 11:10:21,216 INFO [monitor.py:131] Node[0] Epoch[3] Step[600-799] Cost Time: 264.698s Speed: 4.53 samples/sec Remaining Time: 8:45:42 Remaining step percent: 84.66%
2026-07-15 11:10:22,396 INFO [metric_updater.py:360] Node[0] Epoch[3] Step[799] GlobalStep[4318] loss_bevfusion_centerpoint_pointpillar_henet_multisensor_multitask_nuscenes_opt: car_cls_loss[0.4297] car_reg_loss[0.4004] truck_cls_loss[0.5485] truck_reg_loss[0.4274] bus_cls_loss[0.4046] bus_reg_loss[0.4257] barrier_cls_loss[0.4343] barrier_reg_loss[0.4899] bicycle_cls_loss[0.4389] bicycle_reg_loss[0.4527] pedestrian_cls_loss[0.3876] pedestrian_reg_loss[0.5945] loss_occ[1.5779] 
2026-07-15 11:14:40,099 INFO [monitor.py:131] Node[0] Epoch[3] Step[800-999] Cost Time: 258.882s Speed: 4.64 samples/sec Remaining Time: 8:29:50 Remaining step percent: 83.95%
2026-07-15 11:14:41,551 INFO [metric_updater.py:360] Node[0] Epoch[3] Step[999] GlobalStep[4518] loss_bevfusion_centerpoint_pointpillar_henet_multisensor_multitask_nuscenes_opt: car_cls_loss[0.4281] car_reg_loss[0.3996] truck_cls_loss[0.5466] truck_reg_loss[0.4255] bus_cls_loss[0.4093] bus_reg_loss[0.4235] barrier_cls_loss[0.4252] barrier_reg_loss[0.4840] bicycle_cls_loss[0.4426] bicycle_reg_loss[0.4529] pedestrian_cls_loss[0.3878] pedestrian_reg_loss[0.5947] loss_occ[1.5705] 
2026-07-15 11:18:19,009 INFO [monitor.py:146] Node[0] Epoch[3] End   ==================================================
2026-07-15 11:18:19,010 INFO [monitor.py:149] Node[0] Epoch[3] Cost Time: 1545.650s
2026-07-15 11:18:19,010 INFO [metric_updater.py:360] Node[0] Epoch[3] loss_bevfusion_centerpoint_pointpillar_henet_multisensor_multitask_nuscenes_opt: car_cls_loss[0.4279] car_reg_loss[0.3981] truck_cls_loss[0.5480] truck_reg_loss[0.4245] bus_cls_loss[0.4069] bus_reg_loss[0.4219] barrier_cls_loss[0.4326] barrier_reg_loss[0.4838] bicycle_cls_loss[0.4381] bicycle_reg_loss[0.4473] pedestrian_cls_loss[0.3900] pedestrian_reg_loss[0.5950] loss_occ[1.5713] 
2026-07-15 11:18:19,011 INFO [validation.py:156] Node[0] [32mUse train `model` as val model.[0m
2026-07-15 11:18:19,177 INFO [loop_base.py:482] Node[0] [32mStart Predictor loop from epoch 3, num_epochs=1[0m
2026-07-15 11:18:19,180 INFO [fake_quantize.py:253] Node[0] Set fake quantize state to FakeQuantState.VALIDATION
2026-07-15 11:18:25,307 INFO [loop_base.py:553] Node[0] 0 / 1505
2026-07-15 11:18:49,697 INFO [loop_base.py:553] Node[0] 200 / 1505
2026-07-15 11:19:14,714 INFO [loop_base.py:553] Node[0] 400 / 1505
2026-07-15 11:19:38,601 INFO [loop_base.py:553] Node[0] 600 / 1505
2026-07-15 11:20:02,705 INFO [loop_base.py:553] Node[0] 800 / 1505
2026-07-15 11:20:31,877 INFO [loop_base.py:553] Node[0] 1000 / 1505
2026-07-15 11:21:01,380 INFO [loop_base.py:553] Node[0] 1200 / 1505
2026-07-15 11:21:43,006 INFO [loop_base.py:553] Node[0] 1400 / 1505
2026-07-15 11:21:58,170 INFO [nuscenes_metric.py:322] Node[0] The length of self.nusc_annos is: 1505
2026-07-15 11:22:00,506 INFO [nuscenes_metric.py:346] Node[0] Results writes to ./metric_results/bevfusion_centerpoint_pointpillar_henet_multisensor_multitask_nuscenes_opt/results_nusc.json
2026-07-15 11:24:15,199 INFO [nuscenes_metric.py:388] Node[0] NDS: 0.5710, mAP:0.5089
car_AP: [0.5]:0.6660  [1.0]:0.8368  [2.0]:0.8863  [4.0]:0.9044 
truck_AP: [0.5]:0.2936  [1.0]:0.5274  [2.0]:0.6391  [4.0]:0.6759 
trailer_AP: [0.5]:0.0410  [1.0]:0.2840  [2.0]:0.4629  [4.0]:0.5158 
bus_AP: [0.5]:0.3673  [1.0]:0.6421  [2.0]:0.7983  [4.0]:0.8250 
construction_vehicle_AP: [0.5]:0.0021  [1.0]:0.1119  [2.0]:0.2883  [4.0]:0.3609 
bicycle_AP: [0.5]:0.2204  [1.0]:0.3321  [2.0]:0.3447  [4.0]:0.3545 
motorcycle_AP: [0.5]:0.3830  [1.0]:0.6112  [2.0]:0.6413  [4.0]:0.6526 
pedestrian_AP: [0.5]:0.4063  [1.0]:0.5645  [2.0]:0.5911  [4.0]:0.6323 
traffic_cone_AP: [0.5]:0.4566  [1.0]:0.5724  [2.0]:0.6117  [4.0]:0.6644 
barrier_AP: [0.5]:0.2714  [1.0]:0.5801  [2.0]:0.6531  [4.0]:0.6847 

2026-07-15 11:24:15,421 WARNING [metric.py:200] Node[0] <class 'hat.metrics.nuscenes_metric.NuscenesMetric'> not ready for distributed environment, should not be used together with DistributedSampler.Might be slow in validation due to resource competition
2026-07-15 11:24:15,442 INFO [mean_iou.py:170] Node[0] ~~~~ MeanIOU Summary metrics ~~~~
Summary:
Scope                 mIoU       mAcc       aAcc
global               45.23      52.28      70.82
Per Class Results:
Class                  IoU        Acc
others                3.10       3.18
barrier              50.11      63.34
bicycle              22.98      25.41
bus                  64.03      71.20
car                  69.13      77.75
construction_vehicle      29.71      34.70
motorcycle           36.90      41.11
pedestrian           46.37      51.67
traffic_cone         25.34      29.87
trailer              43.77      55.29
truck                50.33      56.66
driveable_surface      77.88      87.86
other_flat           30.99      38.39
sidewalk             42.49      54.26
terrain              49.92      61.23
manmade              61.10      67.30
vegetation           64.70      69.56

2026-07-15 11:24:15,443 INFO [metric_updater.py:360] Node[0] Epoch[3] Validation bevfusion_centerpoint_pointpillar_henet_multisensor_multitask_nuscenes_opt: NDS[0.5710] MeanIOU[tensor(0.4523, device='cuda:0')] 
2026-07-15 11:24:23,154 INFO [checkpoint.py:233] Node[0] Save model checkpoint: ./tmp_models/bevfusion_centerpoint_pointpillar_henet_multisensor_multitask_nuscenes_opt/float-checkpoint-epoch-0003-2957a33b.pth.tar
2026-07-15 11:24:24,053 INFO [checkpoint.py:270] Node[0] Save last model checkpoint: ./tmp_models/bevfusion_centerpoint_pointpillar_henet_multisensor_multitask_nuscenes_opt/float-checkpoint-last-d021d3fa.pth.tar
2026-07-15 11:24:24,062 INFO [monitor.py:143] Node[0] Epoch[4] Begin ==================================================
2026-07-15 11:24:24,062 INFO [lr_updater.py:204] Node[0] Epoch[4] Step[0] GlobalStep[4692] lr=0.000373
2026-07-15 11:29:01,678 INFO [monitor.py:131] Node[0] Epoch[4] Step[0-199] Cost Time: 277.613s Speed: 4.32 samples/sec Remaining Time: 9:00:48 Remaining step percent: 82.62%
2026-07-15 11:29:02,845 INFO [metric_updater.py:360] Node[0] Epoch[4] Step[199] GlobalStep[4891] loss_bevfusion_centerpoint_pointpillar_henet_multisensor_multitask_nuscenes_opt: car_cls_loss[0.4121] car_reg_loss[0.3915] truck_cls_loss[0.5000] truck_reg_loss[0.4093] bus_cls_loss[0.3464] bus_reg_loss[0.3961] barrier_cls_loss[0.4354] barrier_reg_loss[0.4784] bicycle_cls_loss[0.4120] bicycle_reg_loss[0.4281] pedestrian_cls_loss[0.3654] pedestrian_reg_loss[0.5857] loss_occ[1.5536] 
2026-07-15 11:32:57,586 INFO [monitor.py:131] Node[0] Epoch[4] Step[200-399] Cost Time: 235.906s Speed: 5.09 samples/sec Remaining Time: 7:33:19 Remaining step percent: 81.91%
2026-07-15 11:32:58,638 INFO [metric_updater.py:360] Node[0] Epoch[4] Step[399] GlobalStep[5091] loss_bevfusion_centerpoint_pointpillar_henet_multisensor_multitask_nuscenes_opt: car_cls_loss[0.4118] car_reg_loss[0.3915] truck_cls_loss[0.4977] truck_reg_loss[0.4064] bus_cls_loss[0.3587] bus_reg_loss[0.3978] barrier_cls_loss[0.4073] barrier_reg_loss[0.4676] bicycle_cls_loss[0.4111] bicycle_reg_loss[0.4221] pedestrian_cls_loss[0.3686] pedestrian_reg_loss[0.5819] loss_occ[1.5425] 
2026-07-15 11:36:54,548 INFO [monitor.py:131] Node[0] Epoch[4] Step[400-599] Cost Time: 236.959s Speed: 5.06 samples/sec Remaining Time: 7:31:24 Remaining step percent: 81.20%
2026-07-15 11:36:55,558 INFO [metric_updater.py:360] Node[0] Epoch[4] Step[599] GlobalStep[5291] loss_bevfusion_centerpoint_pointpillar_henet_multisensor_multitask_nuscenes_opt: car_cls_loss[0.4106] car_reg_loss[0.3909] truck_cls_loss[0.5060] truck_reg_loss[0.4102] bus_cls_loss[0.3537] bus_reg_loss[0.3960] barrier_cls_loss[0.4016] barrier_reg_loss[0.4624] bicycle_cls_loss[0.4086] bicycle_reg_loss[0.4175] pedestrian_cls_loss[0.3640] pedestrian_reg_loss[0.5807] loss_occ[1.5461] 
2026-07-15 11:40:53,414 INFO [monitor.py:131] Node[0] Epoch[4] Step[600-799] Cost Time: 238.864s Speed: 5.02 samples/sec Remaining Time: 7:31:03 Remaining step percent: 80.49%
2026-07-15 11:40:54,400 INFO [metric_updater.py:360] Node[0] Epoch[4] Step[799] GlobalStep[5491] loss_bevfusion_centerpoint_pointpillar_henet_multisensor_multitask_nuscenes_opt: car_cls_loss[0.4120] car_reg_loss[0.3904] truck_cls_loss[0.5110] truck_reg_loss[0.4123] bus_cls_loss[0.3664] bus_reg_loss[0.3986] barrier_cls_loss[0.3972] barrier_reg_loss[0.4646] bicycle_cls_loss[0.4028] bicycle_reg_loss[0.4165] pedestrian_cls_loss[0.3682] pedestrian_reg_loss[0.5802] loss_occ[1.5418] 
2026-07-15 11:44:53,193 INFO [monitor.py:131] Node[0] Epoch[4] Step[800-999] Cost Time: 239.777s Speed: 5.00 samples/sec Remaining Time: 7:28:46 Remaining step percent: 79.78%
2026-07-15 11:44:54,596 INFO [metric_updater.py:360] Node[0] Epoch[4] Step[999] GlobalStep[5691] loss_bevfusion_centerpoint_pointpillar_henet_multisensor_multitask_nuscenes_opt: car_cls_loss[0.4126] car_reg_loss[0.3899] truck_cls_loss[0.5158] truck_reg_loss[0.4126] bus_cls_loss[0.3701] bus_reg_loss[0.3974] barrier_cls_loss[0.4002] barrier_reg_loss[0.4673] bicycle_cls_loss[0.3998] bicycle_reg_loss[0.4152] pedestrian_cls_loss[0.3683] pedestrian_reg_loss[0.5809] loss_occ[1.5420] 
2026-07-15 11:48:23,471 INFO [monitor.py:146] Node[0] Epoch[4] End   ==================================================
2026-07-15 11:48:23,473 INFO [monitor.py:149] Node[0] Epoch[4] Cost Time: 1439.411s
2026-07-15 11:48:23,473 INFO [metric_updater.py:360] Node[0] Epoch[4] loss_bevfusion_centerpoint_pointpillar_henet_multisensor_multitask_nuscenes_opt: car_cls_loss[0.4119] car_reg_loss[0.3896] truck_cls_loss[0.5234] truck_reg_loss[0.4130] bus_cls_loss[0.3701] bus_reg_loss[0.3967] barrier_cls_loss[0.3952] barrier_reg_loss[0.4626] bicycle_cls_loss[0.3970] bicycle_reg_loss[0.4162] pedestrian_cls_loss[0.3697] pedestrian_reg_loss[0.5817] loss_occ[1.5405] 
2026-07-15 11:48:23,473 INFO [validation.py:156] Node[0] [32mUse train `model` as val model.[0m
2026-07-15 11:48:23,646 INFO [loop_base.py:482] Node[0] [32mStart Predictor loop from epoch 4, num_epochs=1[0m
2026-07-15 11:48:23,649 INFO [fake_quantize.py:253] Node[0] Set fake quantize state to FakeQuantState.VALIDATION
2026-07-15 11:48:30,323 INFO [loop_base.py:553] Node[0] 0 / 1505
2026-07-15 11:49:00,457 INFO [loop_base.py:553] Node[0] 200 / 1505
2026-07-15 11:49:39,500 INFO [loop_base.py:553] Node[0] 400 / 1505
2026-07-15 11:50:07,646 INFO [loop_base.py:553] Node[0] 600 / 1505
2026-07-15 11:50:35,187 INFO [loop_base.py:553] Node[0] 800 / 1505
2026-07-15 11:51:01,655 INFO [loop_base.py:553] Node[0] 1000 / 1505
2026-07-15 11:51:29,529 INFO [loop_base.py:553] Node[0] 1200 / 1505
2026-07-15 11:52:00,893 INFO [loop_base.py:553] Node[0] 1400 / 1505
2026-07-15 11:52:16,605 INFO [nuscenes_metric.py:322] Node[0] The length of self.nusc_annos is: 1505
2026-07-15 11:52:18,820 INFO [nuscenes_metric.py:346] Node[0] Results writes to ./metric_results/bevfusion_centerpoint_pointpillar_henet_multisensor_multitask_nuscenes_opt/results_nusc.json
2026-07-15 11:54:15,043 INFO [nuscenes_metric.py:388] Node[0] NDS: 0.5576, mAP:0.4752
car_AP: [0.5]:0.6312  [1.0]:0.7836  [2.0]:0.8329  [4.0]:0.8524 
truck_AP: [0.5]:0.2665  [1.0]:0.4688  [2.0]:0.5790  [4.0]:0.6191 
trailer_AP: [0.5]:0.0459  [1.0]:0.2151  [2.0]:0.3092  [4.0]:0.3769 
bus_AP: [0.5]:0.3154  [1.0]:0.5327  [2.0]:0.6731  [4.0]:0.7241 
construction_vehicle_AP: [0.5]:0.0027  [1.0]:0.1287  [2.0]:0.2795  [4.0]:0.3478 
bicycle_AP: [0.5]:0.2490  [1.0]:0.3427  [2.0]:0.3701  [4.0]:0.3799 
motorcycle_AP: [0.5]:0.3669  [1.0]:0.5447  [2.0]:0.5791  [4.0]:0.5877 
pedestrian_AP: [0.5]:0.4048  [1.0]:0.5602  [2.0]:0.5872  [4.0]:0.6292 
traffic_cone_AP: [0.5]:0.4585  [1.0]:0.5848  [2.0]:0.6288  [4.0]:0.6847 
barrier_AP: [0.5]:0.2558  [1.0]:0.5427  [2.0]:0.6181  [4.0]:0.6483 

2026-07-15 11:54:15,214 WARNING [metric.py:200] Node[0] <class 'hat.metrics.nuscenes_metric.NuscenesMetric'> not ready for distributed environment, should not be used together with DistributedSampler.Might be slow in validation due to resource competition
2026-07-15 11:54:15,237 INFO [mean_iou.py:170] Node[0] ~~~~ MeanIOU Summary metrics ~~~~
Summary:
Scope                 mIoU       mAcc       aAcc
global               45.33      53.80      70.92
Per Class Results:
Class                  IoU        Acc
others                3.77       3.98
barrier              49.27      61.64
bicycle              23.25      25.23
bus                  63.28      67.07
car                  67.49      74.96
construction_vehicle      28.90      34.43
motorcycle           36.12      38.91
pedestrian           56.04      65.29
traffic_cone         30.70      38.54
trailer              41.15      64.90
truck                51.46      60.67
driveable_surface      74.67      80.17
other_flat           26.09      42.08
sidewalk             41.06      49.54
terrain              48.31      61.82
manmade              62.48      73.29
vegetation           66.66      72.04

2026-07-15 11:54:15,238 INFO [metric_updater.py:360] Node[0] Epoch[4] Validation bevfusion_centerpoint_pointpillar_henet_multisensor_multitask_nuscenes_opt: NDS[0.5576] MeanIOU[tensor(0.4533, device='cuda:0')] 
2026-07-15 11:54:22,264 INFO [checkpoint.py:233] Node[0] Save model checkpoint: ./tmp_models/bevfusion_centerpoint_pointpillar_henet_multisensor_multitask_nuscenes_opt/float-checkpoint-epoch-0004-6dd7c6a6.pth.tar
2026-07-15 11:54:23,110 INFO [checkpoint.py:270] Node[0] Save last model checkpoint: ./tmp_models/bevfusion_centerpoint_pointpillar_henet_multisensor_multitask_nuscenes_opt/float-checkpoint-last-90ee9783.pth.tar
2026-07-15 11:54:23,118 INFO [monitor.py:143] Node[0] Epoch[5] Begin ==================================================
2026-07-15 11:54:23,118 INFO [lr_updater.py:204] Node[0] Epoch[5] Step[0] GlobalStep[5865] lr=0.000359
2026-07-15 11:58:45,166 INFO [monitor.py:131] Node[0] Epoch[5] Step[0-199] Cost Time: 262.047s Speed: 4.58 samples/sec Remaining Time: 8:04:44 Remaining step percent: 78.46%
2026-07-15 11:58:46,292 INFO [metric_updater.py:360] Node[0] Epoch[5] Step[199] GlobalStep[6064] loss_bevfusion_centerpoint_pointpillar_henet_multisensor_multitask_nuscenes_opt: car_cls_loss[0.4005] car_reg_loss[0.3861] truck_cls_loss[0.4542] truck_reg_loss[0.3984] bus_cls_loss[0.3809] bus_reg_loss[0.4104] barrier_cls_loss[0.3863] barrier_reg_loss[0.4704] bicycle_cls_loss[0.3364] bicycle_reg_loss[0.3895] pedestrian_cls_loss[0.3376] pedestrian_reg_loss[0.5697] loss_occ[1.5006] 
2026-07-15 12:02:44,306 INFO [monitor.py:131] Node[0] Epoch[5] Step[200-399] Cost Time: 239.138s Speed: 5.02 samples/sec Remaining Time: 7:16:10 Remaining step percent: 77.75%
2026-07-15 12:02:45,326 INFO [metric_updater.py:360] Node[0] Epoch[5] Step[399] GlobalStep[6264] loss_bevfusion_centerpoint_pointpillar_henet_multisensor_multitask_nuscenes_opt: car_cls_loss[0.4024] car_reg_loss[0.3854] truck_cls_loss[0.4539] truck_reg_loss[0.3969] bus_cls_loss[0.3681] bus_reg_loss[0.3946] barrier_cls_loss[0.3731] barrier_reg_loss[0.4636] bicycle_cls_loss[0.3563] bicycle_reg_loss[0.4067] pedestrian_cls_loss[0.3527] pedestrian_reg_loss[0.5736] loss_occ[1.5043] 
2026-07-15 12:06:37,518 INFO [monitor.py:131] Node[0] Epoch[5] Step[400-599] Cost Time: 233.210s Speed: 5.15 samples/sec Remaining Time: 7:01:28 Remaining step percent: 77.04%
2026-07-15 12:06:38,869 INFO [metric_updater.py:360] Node[0] Epoch[5] Step[599] GlobalStep[6464] loss_bevfusion_centerpoint_pointpillar_henet_multisensor_multitask_nuscenes_opt: car_cls_loss[0.4068] car_reg_loss[0.3854] truck_cls_loss[0.4647] truck_reg_loss[0.3970] bus_cls_loss[0.3637] bus_reg_loss[0.3951] barrier_cls_loss[0.3815] barrier_reg_loss[0.4636] bicycle_cls_loss[0.3796] bicycle_reg_loss[0.4041] pedestrian_cls_loss[0.3629] pedestrian_reg_loss[0.5749] loss_occ[1.5095] 
2026-07-15 12:10:31,529 INFO [monitor.py:131] Node[0] Epoch[5] Step[600-799] Cost Time: 234.009s Speed: 5.13 samples/sec Remaining Time: 6:59:00 Remaining step percent: 76.32%
2026-07-15 12:10:32,549 INFO [metric_updater.py:360] Node[0] Epoch[5] Step[799] GlobalStep[6664] loss_bevfusion_centerpoint_pointpillar_henet_multisensor_multitask_nuscenes_opt: car_cls_loss[0.4068] car_reg_loss[0.3853] truck_cls_loss[0.4734] truck_reg_loss[0.3963] bus_cls_loss[0.3606] bus_reg_loss[0.3954] barrier_cls_loss[0.3889] barrier_reg_loss[0.4613] bicycle_cls_loss[0.3751] bicycle_reg_loss[0.4019] pedestrian_cls_loss[0.3655] pedestrian_reg_loss[0.5748] loss_occ[1.5171] 
2026-07-15 12:14:29,114 INFO [monitor.py:131] Node[0] Epoch[5] Step[800-999] Cost Time: 237.584s Speed: 5.05 samples/sec Remaining Time: 7:01:27 Remaining step percent: 75.61%
2026-07-15 12:14:30,190 INFO [metric_updater.py:360] Node[0] Epoch[5] Step[999] GlobalStep[6864] loss_bevfusion_centerpoint_pointpillar_henet_multisensor_multitask_nuscenes_opt: car_cls_loss[0.4044] car_reg_loss[0.3844] truck_cls_loss[0.4795] truck_reg_loss[0.3990] bus_cls_loss[0.3617] bus_reg_loss[0.3937] barrier_cls_loss[0.3905] barrier_reg_loss[0.4597] bicycle_cls_loss[0.3748] bicycle_reg_loss[0.4015] pedestrian_cls_loss[0.3645] pedestrian_reg_loss[0.5751] loss_occ[1.5141] 
2026-07-15 12:18:00,065 INFO [monitor.py:146] Node[0] Epoch[5] End   ==================================================
2026-07-15 12:18:00,066 INFO [monitor.py:149] Node[0] Epoch[5] Cost Time: 1416.948s
2026-07-15 12:18:00,066 INFO [metric_updater.py:360] Node[0] Epoch[5] loss_bevfusion_centerpoint_pointpillar_henet_multisensor_multitask_nuscenes_opt: car_cls_loss[0.4021] car_reg_loss[0.3837] truck_cls_loss[0.4837] truck_reg_loss[0.3984] bus_cls_loss[0.3585] bus_reg_loss[0.3933] barrier_cls_loss[0.3884] barrier_reg_loss[0.4571] bicycle_cls_loss[0.3635] bicycle_reg_loss[0.4006] pedestrian_cls_loss[0.3649] pedestrian_reg_loss[0.5734] loss_occ[1.5076] 
2026-07-15 12:18:00,066 INFO [validation.py:156] Node[0] [32mUse train `model` as val model.[0m
2026-07-15 12:18:00,238 INFO [loop_base.py:482] Node[0] [32mStart Predictor loop from epoch 5, num_epochs=1[0m
2026-07-15 12:18:00,241 INFO [fake_quantize.py:253] Node[0] Set fake quantize state to FakeQuantState.VALIDATION
2026-07-15 12:18:07,172 INFO [loop_base.py:553] Node[0] 0 / 1505
2026-07-15 12:18:33,439 INFO [loop_base.py:553] Node[0] 200 / 1505
2026-07-15 12:18:57,754 INFO [loop_base.py:553] Node[0] 400 / 1505
2026-07-15 12:19:39,156 INFO [loop_base.py:553] Node[0] 600 / 1505
2026-07-15 12:20:08,843 INFO [loop_base.py:553] Node[0] 800 / 1505
2026-07-15 12:20:53,703 INFO [loop_base.py:553] Node[0] 1000 / 1505
2026-07-15 12:21:37,144 INFO [loop_base.py:553] Node[0] 1200 / 1505
2026-07-15 12:22:02,272 INFO [loop_base.py:553] Node[0] 1400 / 1505
2026-07-15 12:22:15,870 INFO [nuscenes_metric.py:322] Node[0] The length of self.nusc_annos is: 1505
2026-07-15 12:22:18,166 INFO [nuscenes_metric.py:346] Node[0] Results writes to ./metric_results/bevfusion_centerpoint_pointpillar_henet_multisensor_multitask_nuscenes_opt/results_nusc.json
2026-07-15 12:24:25,912 INFO [nuscenes_metric.py:388] Node[0] NDS: 0.5817, mAP:0.5126
car_AP: [0.5]:0.6744  [1.0]:0.8387  [2.0]:0.8875  [4.0]:0.9020 
truck_AP: [0.5]:0.3129  [1.0]:0.5193  [2.0]:0.6263  [4.0]:0.6598 
trailer_AP: [0.5]:0.0385  [1.0]:0.2507  [2.0]:0.4244  [4.0]:0.5413 
bus_AP: [0.5]:0.3766  [1.0]:0.6411  [2.0]:0.7780  [4.0]:0.8075 
construction_vehicle_AP: [0.5]:0.0071  [1.0]:0.1266  [2.0]:0.3173  [4.0]:0.4064 
bicycle_AP: [0.5]:0.2961  [1.0]:0.4039  [2.0]:0.4271  [4.0]:0.4424 
motorcycle_AP: [0.5]:0.3522  [1.0]:0.5665  [2.0]:0.6163  [4.0]:0.6257 
pedestrian_AP: [0.5]:0.3783  [1.0]:0.5618  [2.0]:0.5919  [4.0]:0.6315 
traffic_cone_AP: [0.5]:0.4415  [1.0]:0.5831  [2.0]:0.6283  [4.0]:0.6843 
barrier_AP: [0.5]:0.2831  [1.0]:0.5644  [2.0]:0.6291  [4.0]:0.6602 

2026-07-15 12:24:26,166 WARNING [metric.py:200] Node[0] <class 'hat.metrics.nuscenes_metric.NuscenesMetric'> not ready for distributed environment, should not be used together with DistributedSampler.Might be slow in validation due to resource competition
2026-07-15 12:24:26,187 INFO [mean_iou.py:170] Node[0] ~~~~ MeanIOU Summary metrics ~~~~
Summary:
Scope                 mIoU       mAcc       aAcc
global               47.37      56.07      73.37
Per Class Results:
Class                  IoU        Acc
others                4.64       4.98
barrier              52.68      65.47
bicycle              26.25      30.16
bus                  65.83      69.64
car                  70.90      78.08
construction_vehicle      32.73      38.97
motorcycle           41.38      48.13
pedestrian           53.81      61.27
traffic_cone         29.67      36.66
trailer              44.64      63.53
truck                53.92      63.69
driveable_surface      77.77      83.50
other_flat           31.93      38.19
sidewalk             42.53      53.79
terrain              45.57      70.68
manmade              62.39      68.36
vegetation           68.64      78.05

2026-07-15 12:24:26,189 INFO [metric_updater.py:360] Node[0] Epoch[5] Validation bevfusion_centerpoint_pointpillar_henet_multisensor_multitask_nuscenes_opt: NDS[0.5817] MeanIOU[tensor(0.4737, device='cuda:0')] 
2026-07-15 12:24:35,870 INFO [checkpoint.py:233] Node[0] Save model checkpoint: ./tmp_models/bevfusion_centerpoint_pointpillar_henet_multisensor_multitask_nuscenes_opt/float-checkpoint-epoch-0005-871e7562.pth.tar
2026-07-15 12:24:36,660 INFO [checkpoint.py:270] Node[0] Save last model checkpoint: ./tmp_models/bevfusion_centerpoint_pointpillar_henet_multisensor_multitask_nuscenes_opt/float-checkpoint-last-8f76c352.pth.tar
2026-07-15 12:24:36,671 INFO [monitor.py:143] Node[0] Epoch[6] Begin ==================================================
2026-07-15 12:24:36,672 INFO [lr_updater.py:204] Node[0] Epoch[6] Step[0] GlobalStep[7038] lr=0.000341
2026-07-15 12:28:54,375 INFO [monitor.py:131] Node[0] Epoch[6] Step[0-199] Cost Time: 257.703s Speed: 4.66 samples/sec Remaining Time: 7:31:23 Remaining step percent: 74.29%
2026-07-15 12:28:55,475 INFO [metric_updater.py:360] Node[0] Epoch[6] Step[199] GlobalStep[7237] loss_bevfusion_centerpoint_pointpillar_henet_multisensor_multitask_nuscenes_opt: car_cls_loss[0.3900] car_reg_loss[0.3755] truck_cls_loss[0.4303] truck_reg_loss[0.3884] bus_cls_loss[0.3256] bus_reg_loss[0.3598] barrier_cls_loss[0.3862] barrier_reg_loss[0.4596] bicycle_cls_loss[0.3323] bicycle_reg_loss[0.3923] pedestrian_cls_loss[0.3523] pedestrian_reg_loss[0.5702] loss_occ[1.4760] 
2026-07-15 12:32:47,598 INFO [monitor.py:131] Node[0] Epoch[6] Step[200-399] Cost Time: 233.220s Speed: 5.15 samples/sec Remaining Time: 6:42:34 Remaining step percent: 73.58%
2026-07-15 12:32:48,634 INFO [metric_updater.py:360] Node[0] Epoch[6] Step[399] GlobalStep[7437] loss_bevfusion_centerpoint_pointpillar_henet_multisensor_multitask_nuscenes_opt: car_cls_loss[0.3955] car_reg_loss[0.3746] truck_cls_loss[0.4264] truck_reg_loss[0.3849] bus_cls_loss[0.3378] bus_reg_loss[0.3798] barrier_cls_loss[0.3745] barrier_reg_loss[0.4537] bicycle_cls_loss[0.3314] bicycle_reg_loss[0.3879] pedestrian_cls_loss[0.3496] pedestrian_reg_loss[0.5639] loss_occ[1.4748] 
2026-07-15 12:36:43,912 INFO [monitor.py:131] Node[0] Epoch[6] Step[400-599] Cost Time: 236.313s Speed: 5.08 samples/sec Remaining Time: 6:43:58 Remaining step percent: 72.87%
2026-07-15 12:36:44,937 INFO [metric_updater.py:360] Node[0] Epoch[6] Step[599] GlobalStep[7637] loss_bevfusion_centerpoint_pointpillar_henet_multisensor_multitask_nuscenes_opt: car_cls_loss[0.3958] car_reg_loss[0.3770] truck_cls_loss[0.4430] truck_reg_loss[0.3907] bus_cls_loss[0.3284] bus_reg_loss[0.3797] barrier_cls_loss[0.3714] barrier_reg_loss[0.4542] bicycle_cls_loss[0.3251] bicycle_reg_loss[0.3864] pedestrian_cls_loss[0.3519] pedestrian_reg_loss[0.5643] loss_occ[1.4842] 
2026-07-15 12:40:37,295 INFO [monitor.py:131] Node[0] Epoch[6] Step[600-799] Cost Time: 233.381s Speed: 5.14 samples/sec Remaining Time: 6:35:04 Remaining step percent: 72.16%
2026-07-15 12:40:38,273 INFO [metric_updater.py:360] Node[0] Epoch[6] Step[799] GlobalStep[7837] loss_bevfusion_centerpoint_pointpillar_henet_multisensor_multitask_nuscenes_opt: car_cls_loss[0.3970] car_reg_loss[0.3761] truck_cls_loss[0.4476] truck_reg_loss[0.3906] bus_cls_loss[0.3376] bus_reg_loss[0.3780] barrier_cls_loss[0.3650] barrier_reg_loss[0.4491] bicycle_cls_loss[0.3296] bicycle_reg_loss[0.3829] pedestrian_cls_loss[0.3513] pedestrian_reg_loss[0.5632] loss_occ[1.4801] 
2026-07-15 12:44:30,961 INFO [monitor.py:131] Node[0] Epoch[6] Step[800-999] Cost Time: 233.663s Speed: 5.14 samples/sec Remaining Time: 6:31:39 Remaining step percent: 71.45%
2026-07-15 12:44:32,003 INFO [metric_updater.py:360] Node[0] Epoch[6] Step[999] GlobalStep[8037] loss_bevfusion_centerpoint_pointpillar_henet_multisensor_multitask_nuscenes_opt: car_cls_loss[0.3964] car_reg_loss[0.3764] truck_cls_loss[0.4480] truck_reg_loss[0.3906] bus_cls_loss[0.3395] bus_reg_loss[0.3803] barrier_cls_loss[0.3596] barrier_reg_loss[0.4418] bicycle_cls_loss[0.3183] bicycle_reg_loss[0.3811] pedestrian_cls_loss[0.3548] pedestrian_reg_loss[0.5633] loss_occ[1.4853] 
2026-07-15 12:48:00,336 INFO [monitor.py:146] Node[0] Epoch[6] End   ==================================================
2026-07-15 12:48:00,338 INFO [monitor.py:149] Node[0] Epoch[6] Cost Time: 1403.666s
2026-07-15 12:48:00,338 INFO [metric_updater.py:360] Node[0] Epoch[6] loss_bevfusion_centerpoint_pointpillar_henet_multisensor_multitask_nuscenes_opt: car_cls_loss[0.3964] car_reg_loss[0.3771] truck_cls_loss[0.4462] truck_reg_loss[0.3906] bus_cls_loss[0.3338] bus_reg_loss[0.3780] barrier_cls_loss[0.3613] barrier_reg_loss[0.4427] bicycle_cls_loss[0.3177] bicycle_reg_loss[0.3821] pedestrian_cls_loss[0.3536] pedestrian_reg_loss[0.5643] loss_occ[1.4817] 
2026-07-15 12:48:00,339 INFO [validation.py:156] Node[0] [32mUse train `model` as val model.[0m
2026-07-15 12:48:00,522 INFO [loop_base.py:482] Node[0] [32mStart Predictor loop from epoch 6, num_epochs=1[0m
2026-07-15 12:48:00,525 INFO [fake_quantize.py:253] Node[0] Set fake quantize state to FakeQuantState.VALIDATION
2026-07-15 12:48:07,442 INFO [loop_base.py:553] Node[0] 0 / 1505
2026-07-15 12:48:32,699 INFO [loop_base.py:553] Node[0] 200 / 1505
2026-07-15 12:49:23,025 INFO [loop_base.py:553] Node[0] 400 / 1505
2026-07-15 12:50:02,577 INFO [loop_base.py:553] Node[0] 600 / 1505
2026-07-15 12:50:35,873 INFO [loop_base.py:553] Node[0] 800 / 1505
2026-07-15 12:51:13,848 INFO [loop_base.py:553] Node[0] 1000 / 1505
2026-07-15 12:51:48,553 INFO [loop_base.py:553] Node[0] 1200 / 1505
2026-07-15 12:52:27,998 INFO [loop_base.py:553] Node[0] 1400 / 1505
2026-07-15 12:52:41,405 INFO [nuscenes_metric.py:322] Node[0] The length of self.nusc_annos is: 1505
2026-07-15 12:52:43,409 INFO [nuscenes_metric.py:346] Node[0] Results writes to ./metric_results/bevfusion_centerpoint_pointpillar_henet_multisensor_multitask_nuscenes_opt/results_nusc.json
2026-07-15 12:54:39,948 INFO [nuscenes_metric.py:388] Node[0] NDS: 0.5877, mAP:0.5189
car_AP: [0.5]:0.6826  [1.0]:0.8405  [2.0]:0.8889  [4.0]:0.9038 
truck_AP: [0.5]:0.3238  [1.0]:0.5229  [2.0]:0.6391  [4.0]:0.6756 
trailer_AP: [0.5]:0.0662  [1.0]:0.2678  [2.0]:0.4231  [4.0]:0.5244 
bus_AP: [0.5]:0.4086  [1.0]:0.6799  [2.0]:0.8004  [4.0]:0.8306 
construction_vehicle_AP: [0.5]:0.0031  [1.0]:0.1207  [2.0]:0.2938  [4.0]:0.3751 
bicycle_AP: [0.5]:0.2756  [1.0]:0.3894  [2.0]:0.4066  [4.0]:0.4226 
motorcycle_AP: [0.5]:0.4018  [1.0]:0.6139  [2.0]:0.6501  [4.0]:0.6661 
pedestrian_AP: [0.5]:0.3776  [1.0]:0.5591  [2.0]:0.5931  [4.0]:0.6357 
traffic_cone_AP: [0.5]:0.4560  [1.0]:0.5767  [2.0]:0.6164  [4.0]:0.6717 
barrier_AP: [0.5]:0.3039  [1.0]:0.5687  [2.0]:0.6392  [4.0]:0.6621 

2026-07-15 12:54:40,136 WARNING [metric.py:200] Node[0] <class 'hat.metrics.nuscenes_metric.NuscenesMetric'> not ready for distributed environment, should not be used together with DistributedSampler.Might be slow in validation due to resource competition
2026-07-15 12:54:40,154 INFO [mean_iou.py:170] Node[0] ~~~~ MeanIOU Summary metrics ~~~~
Summary:
Scope                 mIoU       mAcc       aAcc
global               45.75      52.37      73.45
Per Class Results:
Class                  IoU        Acc
others                3.89       4.00
barrier              49.00      57.94
bicycle              25.71      28.42
bus                  63.10      66.65
car                  68.01      76.13
construction_vehicle      26.97      29.76
motorcycle           37.71      41.26
pedestrian           48.39      54.16
traffic_cone         21.86      24.01
trailer              44.52      52.09
truck                53.14      60.82
driveable_surface      79.00      85.60
other_flat           31.41      36.14
sidewalk             44.86      59.21
terrain              47.34      69.07
manmade              63.25      68.38
vegetation           69.53      76.62

2026-07-15 12:54:40,156 INFO [metric_updater.py:360] Node[0] Epoch[6] Validation bevfusion_centerpoint_pointpillar_henet_multisensor_multitask_nuscenes_opt: NDS[0.5877] MeanIOU[tensor(0.4575, device='cuda:0')] 
2026-07-15 12:54:45,738 INFO [checkpoint.py:233] Node[0] Save model checkpoint: ./tmp_models/bevfusion_centerpoint_pointpillar_henet_multisensor_multitask_nuscenes_opt/float-checkpoint-epoch-0006-99b98713.pth.tar
2026-07-15 12:54:46,567 INFO [checkpoint.py:270] Node[0] Save last model checkpoint: ./tmp_models/bevfusion_centerpoint_pointpillar_henet_multisensor_multitask_nuscenes_opt/float-checkpoint-last-2cb1e1a2.pth.tar
2026-07-15 12:54:46,575 INFO [monitor.py:143] Node[0] Epoch[7] Begin ==================================================
2026-07-15 12:54:46,575 INFO [lr_updater.py:204] Node[0] Epoch[7] Step[0] GlobalStep[8211] lr=0.000322
2026-07-15 12:59:07,749 INFO [monitor.py:131] Node[0] Epoch[7] Step[0-199] Cost Time: 261.173s Speed: 4.59 samples/sec Remaining Time: 7:11:48 Remaining step percent: 70.12%
2026-07-15 12:59:09,394 INFO [metric_updater.py:360] Node[0] Epoch[7] Step[199] GlobalStep[8410] loss_bevfusion_centerpoint_pointpillar_henet_multisensor_multitask_nuscenes_opt: car_cls_loss[0.3925] car_reg_loss[0.3728] truck_cls_loss[0.4146] truck_reg_loss[0.3810] bus_cls_loss[0.2710] bus_reg_loss[0.3488] barrier_cls_loss[0.3596] barrier_reg_loss[0.4531] bicycle_cls_loss[0.2627] bicycle_reg_loss[0.3668] pedestrian_cls_loss[0.3298] pedestrian_reg_loss[0.5542] loss_occ[1.4413] 
2026-07-15 13:03:01,653 INFO [monitor.py:131] Node[0] Epoch[7] Step[200-399] Cost Time: 233.899s Speed: 5.13 samples/sec Remaining Time: 6:20:53 Remaining step percent: 69.41%
2026-07-15 13:03:02,615 INFO [metric_updater.py:360] Node[0] Epoch[7] Step[399] GlobalStep[8610] loss_bevfusion_centerpoint_pointpillar_henet_multisensor_multitask_nuscenes_opt: car_cls_loss[0.3846] car_reg_loss[0.3713] truck_cls_loss[0.4133] truck_reg_loss[0.3789] bus_cls_loss[0.2816] bus_reg_loss[0.3533] barrier_cls_loss[0.3501] barrier_reg_loss[0.4455] bicycle_cls_loss[0.2915] bicycle_reg_loss[0.3778] pedestrian_cls_loss[0.3375] pedestrian_reg_loss[0.5568] loss_occ[1.4509] 
2026-07-15 13:06:51,813 INFO [monitor.py:131] Node[0] Epoch[7] Step[400-599] Cost Time: 230.158s Speed: 5.21 samples/sec Remaining Time: 6:10:57 Remaining step percent: 68.70%
2026-07-15 13:06:52,909 INFO [metric_updater.py:360] Node[0] Epoch[7] Step[599] GlobalStep[8810] loss_bevfusion_centerpoint_pointpillar_henet_multisensor_multitask_nuscenes_opt: car_cls_loss[0.3800] car_reg_loss[0.3700] truck_cls_loss[0.4128] truck_reg_loss[0.3791] bus_cls_loss[0.2845] bus_reg_loss[0.3548] barrier_cls_loss[0.3391] barrier_reg_loss[0.4364] bicycle_cls_loss[0.2935] bicycle_reg_loss[0.3739] pedestrian_cls_loss[0.3352] pedestrian_reg_loss[0.5575] loss_occ[1.4461] 
2026-07-15 13:10:48,243 INFO [monitor.py:131] Node[0] Epoch[7] Step[600-799] Cost Time: 236.428s Speed: 5.08 samples/sec Remaining Time: 6:17:07 Remaining step percent: 67.99%
2026-07-15 13:10:49,383 INFO [metric_updater.py:360] Node[0] Epoch[7] Step[799] GlobalStep[9010] loss_bevfusion_centerpoint_pointpillar_henet_multisensor_multitask_nuscenes_opt: car_cls_loss[0.3800] car_reg_loss[0.3709] truck_cls_loss[0.4138] truck_reg_loss[0.3789] bus_cls_loss[0.2837] bus_reg_loss[0.3573] barrier_cls_loss[0.3299] barrier_reg_loss[0.4252] bicycle_cls_loss[0.2916] bicycle_reg_loss[0.3710] pedestrian_cls_loss[0.3365] pedestrian_reg_loss[0.5561] loss_occ[1.4451] 
2026-07-15 13:14:41,963 INFO [monitor.py:131] Node[0] Epoch[7] Step[800-999] Cost Time: 233.718s Speed: 5.13 samples/sec Remaining Time: 6:08:54 Remaining step percent: 67.28%
2026-07-15 13:14:42,958 INFO [metric_updater.py:360] Node[0] Epoch[7] Step[999] GlobalStep[9210] loss_bevfusion_centerpoint_pointpillar_henet_multisensor_multitask_nuscenes_opt: car_cls_loss[0.3807] car_reg_loss[0.3718] truck_cls_loss[0.4144] truck_reg_loss[0.3779] bus_cls_loss[0.2871] bus_reg_loss[0.3555] barrier_cls_loss[0.3382] barrier_reg_loss[0.4262] bicycle_cls_loss[0.2915] bicycle_reg_loss[0.3723] pedestrian_cls_loss[0.3361] pedestrian_reg_loss[0.5561] loss_occ[1.4459] 
2026-07-15 13:18:15,193 INFO [monitor.py:146] Node[0] Epoch[7] End   ==================================================
2026-07-15 13:18:15,195 INFO [monitor.py:149] Node[0] Epoch[7] Cost Time: 1408.620s
2026-07-15 13:18:15,195 INFO [metric_updater.py:360] Node[0] Epoch[7] loss_bevfusion_centerpoint_pointpillar_henet_multisensor_multitask_nuscenes_opt: car_cls_loss[0.3802] car_reg_loss[0.3710] truck_cls_loss[0.4198] truck_reg_loss[0.3800] bus_cls_loss[0.2888] bus_reg_loss[0.3546] barrier_cls_loss[0.3334] barrier_reg_loss[0.4247] bicycle_cls_loss[0.2881] bicycle_reg_loss[0.3717] pedestrian_cls_loss[0.3339] pedestrian_reg_loss[0.5550] loss_occ[1.4455] 
2026-07-15 13:18:15,196 INFO [validation.py:156] Node[0] [32mUse train `model` as val model.[0m
2026-07-15 13:18:15,369 INFO [loop_base.py:482] Node[0] [32mStart Predictor loop from epoch 7, num_epochs=1[0m
2026-07-15 13:18:15,372 INFO [fake_quantize.py:253] Node[0] Set fake quantize state to FakeQuantState.VALIDATION
2026-07-15 13:18:22,083 INFO [loop_base.py:553] Node[0] 0 / 1505
2026-07-15 13:18:49,749 INFO [loop_base.py:553] Node[0] 200 / 1505
2026-07-15 13:19:43,741 INFO [loop_base.py:553] Node[0] 400 / 1505
2026-07-15 13:20:09,354 INFO [loop_base.py:553] Node[0] 600 / 1505
2026-07-15 13:20:34,965 INFO [loop_base.py:553] Node[0] 800 / 1505
2026-07-15 13:21:00,946 INFO [loop_base.py:553] Node[0] 1000 / 1505
2026-07-15 13:21:27,289 INFO [loop_base.py:553] Node[0] 1200 / 1505
2026-07-15 13:21:52,807 INFO [loop_base.py:553] Node[0] 1400 / 1505
2026-07-15 13:22:07,533 INFO [nuscenes_metric.py:322] Node[0] The length of self.nusc_annos is: 1505
2026-07-15 13:22:09,844 INFO [nuscenes_metric.py:346] Node[0] Results writes to ./metric_results/bevfusion_centerpoint_pointpillar_henet_multisensor_multitask_nuscenes_opt/results_nusc.json
2026-07-15 13:24:10,562 INFO [nuscenes_metric.py:388] Node[0] NDS: 0.5707, mAP:0.4925
car_AP: [0.5]:0.6632  [1.0]:0.8182  [2.0]:0.8627  [4.0]:0.8832 
truck_AP: [0.5]:0.2954  [1.0]:0.5047  [2.0]:0.5972  [4.0]:0.6300 
trailer_AP: [0.5]:0.0546  [1.0]:0.2185  [2.0]:0.3176  [4.0]:0.4066 
bus_AP: [0.5]:0.3683  [1.0]:0.6200  [2.0]:0.7369  [4.0]:0.7732 
construction_vehicle_AP: [0.5]:0.0042  [1.0]:0.1190  [2.0]:0.2976  [4.0]:0.3831 
bicycle_AP: [0.5]:0.2763  [1.0]:0.3635  [2.0]:0.3734  [4.0]:0.3821 
motorcycle_AP: [0.5]:0.3629  [1.0]:0.5529  [2.0]:0.5884  [4.0]:0.5985 
pedestrian_AP: [0.5]:0.4088  [1.0]:0.5669  [2.0]:0.5939  [4.0]:0.6290 
traffic_cone_AP: [0.5]:0.4624  [1.0]:0.5825  [2.0]:0.6240  [4.0]:0.6773 
barrier_AP: [0.5]:0.2973  [1.0]:0.5526  [2.0]:0.6151  [4.0]:0.6392 

2026-07-15 13:24:10,731 WARNING [metric.py:200] Node[0] <class 'hat.metrics.nuscenes_metric.NuscenesMetric'> not ready for distributed environment, should not be used together with DistributedSampler.Might be slow in validation due to resource competition
2026-07-15 13:24:10,748 INFO [mean_iou.py:170] Node[0] ~~~~ MeanIOU Summary metrics ~~~~
Summary:
Scope                 mIoU       mAcc       aAcc
global               47.31      56.23      73.40
Per Class Results:
Class                  IoU        Acc
others                4.18       4.35
barrier              51.05      64.24
bicycle              27.71      31.29
bus                  66.99      75.07
car                  70.20      78.54
construction_vehicle      31.04      35.92
motorcycle           35.56      37.97
pedestrian           54.80      62.95
traffic_cone         30.69      38.81
trailer              43.58      58.25
truck                55.01      65.41
driveable_surface      77.49      82.27
other_flat           31.32      47.14
sidewalk             43.72      55.71
terrain              49.07      71.40
manmade              63.59      70.54
vegetation           68.29      76.04

2026-07-15 13:24:10,749 INFO [metric_updater.py:360] Node[0] Epoch[7] Validation bevfusion_centerpoint_pointpillar_henet_multisensor_multitask_nuscenes_opt: NDS[0.5707] MeanIOU[tensor(0.4731, device='cuda:0')] 
2026-07-15 13:24:18,442 INFO [checkpoint.py:233] Node[0] Save model checkpoint: ./tmp_models/bevfusion_centerpoint_pointpillar_henet_multisensor_multitask_nuscenes_opt/float-checkpoint-epoch-0007-8c6b876e.pth.tar
2026-07-15 13:24:19,410 INFO [checkpoint.py:270] Node[0] Save last model checkpoint: ./tmp_models/bevfusion_centerpoint_pointpillar_henet_multisensor_multitask_nuscenes_opt/float-checkpoint-last-23da37ff.pth.tar
2026-07-15 13:24:19,419 INFO [monitor.py:143] Node[0] Epoch[8] Begin ==================================================
2026-07-15 13:24:19,419 INFO [lr_updater.py:204] Node[0] Epoch[8] Step[0] GlobalStep[9384] lr=0.000300
2026-07-15 13:28:39,809 INFO [monitor.py:131] Node[0] Epoch[8] Step[0-199] Cost Time: 260.389s Speed: 4.61 samples/sec Remaining Time: 6:44:56 Remaining step percent: 65.96%
2026-07-15 13:28:40,946 INFO [metric_updater.py:360] Node[0] Epoch[8] Step[199] GlobalStep[9583] loss_bevfusion_centerpoint_pointpillar_henet_multisensor_multitask_nuscenes_opt: car_cls_loss[0.3774] car_reg_loss[0.3680] truck_cls_loss[0.3834] truck_reg_loss[0.3596] bus_cls_loss[0.2876] bus_reg_loss[0.3594] barrier_cls_loss[0.3144] barrier_reg_loss[0.4204] bicycle_cls_loss[0.2257] bicycle_reg_loss[0.3578] pedestrian_cls_loss[0.3227] pedestrian_reg_loss[0.5485] loss_occ[1.4225] 
2026-07-15 13:32:34,994 INFO [monitor.py:131] Node[0] Epoch[8] Step[200-399] Cost Time: 235.184s Speed: 5.10 samples/sec Remaining Time: 5:59:59 Remaining step percent: 65.25%
2026-07-15 13:32:36,213 INFO [metric_updater.py:360] Node[0] Epoch[8] Step[399] GlobalStep[9783] loss_bevfusion_centerpoint_pointpillar_henet_multisensor_multitask_nuscenes_opt: car_cls_loss[0.3772] car_reg_loss[0.3663] truck_cls_loss[0.3740] truck_reg_loss[0.3576] bus_cls_loss[0.2805] bus_reg_loss[0.3594] barrier_cls_loss[0.3126] barrier_reg_loss[0.4182] bicycle_cls_loss[0.2458] bicycle_reg_loss[0.3587] pedestrian_cls_loss[0.3171] pedestrian_reg_loss[0.5476] loss_occ[1.4370] 
2026-07-15 13:36:27,663 INFO [monitor.py:131] Node[0] Epoch[8] Step[400-599] Cost Time: 232.668s Speed: 5.16 samples/sec Remaining Time: 5:52:15 Remaining step percent: 64.54%
2026-07-15 13:36:28,762 INFO [metric_updater.py:360] Node[0] Epoch[8] Step[599] GlobalStep[9983] loss_bevfusion_centerpoint_pointpillar_henet_multisensor_multitask_nuscenes_opt: car_cls_loss[0.3761] car_reg_loss[0.3661] truck_cls_loss[0.3798] truck_reg_loss[0.3620] bus_cls_loss[0.2748] bus_reg_loss[0.3589] barrier_cls_loss[0.3164] barrier_reg_loss[0.4191] bicycle_cls_loss[0.2560] bicycle_reg_loss[0.3592] pedestrian_cls_loss[0.3203] pedestrian_reg_loss[0.5467] loss_occ[1.4351] 
2026-07-15 13:40:19,096 INFO [monitor.py:131] Node[0] Epoch[8] Step[600-799] Cost Time: 231.432s Speed: 5.19 samples/sec Remaining Time: 5:46:31 Remaining step percent: 63.82%
2026-07-15 13:40:20,142 INFO [metric_updater.py:360] Node[0] Epoch[8] Step[799] GlobalStep[10183] loss_bevfusion_centerpoint_pointpillar_henet_multisensor_multitask_nuscenes_opt: car_cls_loss[0.3765] car_reg_loss[0.3678] truck_cls_loss[0.3793] truck_reg_loss[0.3642] bus_cls_loss[0.2680] bus_reg_loss[0.3539] barrier_cls_loss[0.3278] barrier_reg_loss[0.4226] bicycle_cls_loss[0.2594] bicycle_reg_loss[0.3582] pedestrian_cls_loss[0.3219] pedestrian_reg_loss[0.5469] loss_occ[1.4325] 
2026-07-15 13:44:17,727 INFO [monitor.py:131] Node[0] Epoch[8] Step[800-999] Cost Time: 238.630s Speed: 5.03 samples/sec Remaining Time: 5:53:19 Remaining step percent: 63.11%
2026-07-15 13:44:18,753 INFO [metric_updater.py:360] Node[0] Epoch[8] Step[999] GlobalStep[10383] loss_bevfusion_centerpoint_pointpillar_henet_multisensor_multitask_nuscenes_opt: car_cls_loss[0.3787] car_reg_loss[0.3679] truck_cls_loss[0.3831] truck_reg_loss[0.3653] bus_cls_loss[0.2727] bus_reg_loss[0.3539] barrier_cls_loss[0.3284] barrier_reg_loss[0.4227] bicycle_cls_loss[0.2633] bicycle_reg_loss[0.3593] pedestrian_cls_loss[0.3218] pedestrian_reg_loss[0.5467] loss_occ[1.4351] 
2026-07-15 13:47:48,876 INFO [monitor.py:146] Node[0] Epoch[8] End   ==================================================
2026-07-15 13:47:48,878 INFO [monitor.py:149] Node[0] Epoch[8] Cost Time: 1409.459s
2026-07-15 13:47:48,879 INFO [metric_updater.py:360] Node[0] Epoch[8] loss_bevfusion_centerpoint_pointpillar_henet_multisensor_multitask_nuscenes_opt: car_cls_loss[0.3780] car_reg_loss[0.3670] truck_cls_loss[0.3856] truck_reg_loss[0.3650] bus_cls_loss[0.2719] bus_reg_loss[0.3492] barrier_cls_loss[0.3265] barrier_reg_loss[0.4210] bicycle_cls_loss[0.2642] bicycle_reg_loss[0.3587] pedestrian_cls_loss[0.3229] pedestrian_reg_loss[0.5452] loss_occ[1.4353] 
2026-07-15 13:47:48,880 INFO [validation.py:156] Node[0] [32mUse train `model` as val model.[0m
2026-07-15 13:47:49,101 INFO [loop_base.py:482] Node[0] [32mStart Predictor loop from epoch 8, num_epochs=1[0m
2026-07-15 13:47:49,106 INFO [fake_quantize.py:253] Node[0] Set fake quantize state to FakeQuantState.VALIDATION
2026-07-15 13:47:55,831 INFO [loop_base.py:553] Node[0] 0 / 1505
2026-07-15 13:48:33,244 INFO [loop_base.py:553] Node[0] 200 / 1505
2026-07-15 13:49:28,707 INFO [loop_base.py:553] Node[0] 400 / 1505
2026-07-15 13:50:05,282 INFO [loop_base.py:553] Node[0] 600 / 1505
2026-07-15 13:50:29,667 INFO [loop_base.py:553] Node[0] 800 / 1505
2026-07-15 13:51:01,531 INFO [loop_base.py:553] Node[0] 1000 / 1505
2026-07-15 13:51:26,575 INFO [loop_base.py:553] Node[0] 1200 / 1505
2026-07-15 13:52:05,156 INFO [loop_base.py:553] Node[0] 1400 / 1505
2026-07-15 13:52:19,540 INFO [nuscenes_metric.py:322] Node[0] The length of self.nusc_annos is: 1505
2026-07-15 13:52:21,622 INFO [nuscenes_metric.py:346] Node[0] Results writes to ./metric_results/bevfusion_centerpoint_pointpillar_henet_multisensor_multitask_nuscenes_opt/results_nusc.json
2026-07-15 13:54:26,051 INFO [nuscenes_metric.py:388] Node[0] NDS: 0.5847, mAP:0.5161
car_AP: [0.5]:0.6803  [1.0]:0.8436  [2.0]:0.8919  [4.0]:0.9101 
truck_AP: [0.5]:0.3099  [1.0]:0.5288  [2.0]:0.6421  [4.0]:0.6765 
trailer_AP: [0.5]:0.0739  [1.0]:0.3024  [2.0]:0.4443  [4.0]:0.5303 
bus_AP: [0.5]:0.4041  [1.0]:0.6551  [2.0]:0.7952  [4.0]:0.8252 
construction_vehicle_AP: [0.5]:0.0040  [1.0]:0.1057  [2.0]:0.2482  [4.0]:0.3202 
bicycle_AP: [0.5]:0.3391  [1.0]:0.4345  [2.0]:0.4478  [4.0]:0.4596 
motorcycle_AP: [0.5]:0.3766  [1.0]:0.5932  [2.0]:0.6097  [4.0]:0.6203 
pedestrian_AP: [0.5]:0.3936  [1.0]:0.5627  [2.0]:0.5948  [4.0]:0.6302 
traffic_cone_AP: [0.5]:0.4562  [1.0]:0.5744  [2.0]:0.6166  [4.0]:0.6711 
barrier_AP: [0.5]:0.2816  [1.0]:0.5461  [2.0]:0.6127  [4.0]:0.6297 

2026-07-15 13:54:26,208 WARNING [metric.py:200] Node[0] <class 'hat.metrics.nuscenes_metric.NuscenesMetric'> not ready for distributed environment, should not be used together with DistributedSampler.Might be slow in validation due to resource competition
2026-07-15 13:54:26,231 INFO [mean_iou.py:170] Node[0] ~~~~ MeanIOU Summary metrics ~~~~
Summary:
Scope                 mIoU       mAcc       aAcc
global               47.60      54.18      75.66
Per Class Results:
Class                  IoU        Acc
others                4.35       4.59
barrier              48.60      56.21
bicycle              27.87      31.45
bus                  64.66      68.17
car                  69.12      78.14
construction_vehicle      25.16      27.59
motorcycle           36.66      39.08
pedestrian           51.70      57.13
traffic_cone         28.47      33.06
trailer              47.02      57.54
truck                54.69      62.38
driveable_surface      80.56      88.14
other_flat           34.12      39.48
sidewalk             45.67      57.68
terrain              53.68      68.96
manmade              67.99      76.46
vegetation           68.96      75.07

2026-07-15 13:54:26,232 INFO [metric_updater.py:360] Node[0] Epoch[8] Validation bevfusion_centerpoint_pointpillar_henet_multisensor_multitask_nuscenes_opt: NDS[0.5847] MeanIOU[tensor(0.4760, device='cuda:0')] 
2026-07-15 13:54:33,770 INFO [checkpoint.py:233] Node[0] Save model checkpoint: ./tmp_models/bevfusion_centerpoint_pointpillar_henet_multisensor_multitask_nuscenes_opt/float-checkpoint-epoch-0008-7dba7ed3.pth.tar
2026-07-15 13:54:34,702 INFO [checkpoint.py:270] Node[0] Save last model checkpoint: ./tmp_models/bevfusion_centerpoint_pointpillar_henet_multisensor_multitask_nuscenes_opt/float-checkpoint-last-4e3ea55a.pth.tar
2026-07-15 13:54:34,712 INFO [monitor.py:143] Node[0] Epoch[9] Begin ==================================================
2026-07-15 13:54:34,713 INFO [lr_updater.py:204] Node[0] Epoch[9] Step[0] GlobalStep[10557] lr=0.000277
2026-07-15 13:59:31,683 INFO [monitor.py:131] Node[0] Epoch[9] Step[0-199] Cost Time: 296.970s Speed: 4.04 samples/sec Remaining Time: 7:12:38 Remaining step percent: 61.79%
2026-07-15 13:59:32,849 INFO [metric_updater.py:360] Node[0] Epoch[9] Step[199] GlobalStep[10756] loss_bevfusion_centerpoint_pointpillar_henet_multisensor_multitask_nuscenes_opt: car_cls_loss[0.3774] car_reg_loss[0.3714] truck_cls_loss[0.3661] truck_reg_loss[0.3596] bus_cls_loss[0.2451] bus_reg_loss[0.3238] barrier_cls_loss[0.3001] barrier_reg_loss[0.4205] bicycle_cls_loss[0.2535] bicycle_reg_loss[0.3363] pedestrian_cls_loss[0.3149] pedestrian_reg_loss[0.5421] loss_occ[1.4290] 
2026-07-15 14:03:29,682 INFO [monitor.py:131] Node[0] Epoch[9] Step[200-399] Cost Time: 237.998s Speed: 5.04 samples/sec Remaining Time: 5:41:01 Remaining step percent: 61.08%
2026-07-15 14:03:30,854 INFO [metric_updater.py:360] Node[0] Epoch[9] Step[399] GlobalStep[10956] loss_bevfusion_centerpoint_pointpillar_henet_multisensor_multitask_nuscenes_opt: car_cls_loss[0.3692] car_reg_loss[0.3651] truck_cls_loss[0.3644] truck_reg_loss[0.3613] bus_cls_loss[0.2459] bus_reg_loss[0.3323] barrier_cls_loss[0.3166] barrier_reg_loss[0.4277] bicycle_cls_loss[0.2382] bicycle_reg_loss[0.3306] pedestrian_cls_loss[0.3075] pedestrian_reg_loss[0.5436] loss_occ[1.4085] 
2026-07-15 14:07:26,121 INFO [monitor.py:131] Node[0] Epoch[9] Step[400-599] Cost Time: 236.438s Speed: 5.08 samples/sec Remaining Time: 5:34:51 Remaining step percent: 60.37%
2026-07-15 14:07:27,408 INFO [metric_updater.py:360] Node[0] Epoch[9] Step[599] GlobalStep[11156] loss_bevfusion_centerpoint_pointpillar_henet_multisensor_multitask_nuscenes_opt: car_cls_loss[0.3666] car_reg_loss[0.3637] truck_cls_loss[0.3604] truck_reg_loss[0.3589] bus_cls_loss[0.2554] bus_reg_loss[0.3330] barrier_cls_loss[0.3276] barrier_reg_loss[0.4223] bicycle_cls_loss[0.2423] bicycle_reg_loss[0.3373] pedestrian_cls_loss[0.3052] pedestrian_reg_loss[0.5423] loss_occ[1.4056] 
2026-07-15 14:11:26,397 INFO [monitor.py:131] Node[0] Epoch[9] Step[600-799] Cost Time: 240.275s Speed: 4.99 samples/sec Remaining Time: 5:36:17 Remaining step percent: 59.66%
2026-07-15 14:11:27,480 INFO [metric_updater.py:360] Node[0] Epoch[9] Step[799] GlobalStep[11356] loss_bevfusion_centerpoint_pointpillar_henet_multisensor_multitask_nuscenes_opt: car_cls_loss[0.3700] car_reg_loss[0.3641] truck_cls_loss[0.3621] truck_reg_loss[0.3583] bus_cls_loss[0.2588] bus_reg_loss[0.3339] barrier_cls_loss[0.3234] barrier_reg_loss[0.4201] bicycle_cls_loss[0.2525] bicycle_reg_loss[0.3439] pedestrian_cls_loss[0.3081] pedestrian_reg_loss[0.5414] loss_occ[1.4023] 
2026-07-15 14:15:27,853 INFO [monitor.py:131] Node[0] Epoch[9] Step[800-999] Cost Time: 241.454s Speed: 4.97 samples/sec Remaining Time: 5:33:54 Remaining step percent: 58.95%
2026-07-15 14:15:29,143 INFO [metric_updater.py:360] Node[0] Epoch[9] Step[999] GlobalStep[11556] loss_bevfusion_centerpoint_pointpillar_henet_multisensor_multitask_nuscenes_opt: car_cls_loss[0.3696] car_reg_loss[0.3639] truck_cls_loss[0.3666] truck_reg_loss[0.3589] bus_cls_loss[0.2666] bus_reg_loss[0.3389] barrier_cls_loss[0.3292] barrier_reg_loss[0.4241] bicycle_cls_loss[0.2516] bicycle_reg_loss[0.3466] pedestrian_cls_loss[0.3110] pedestrian_reg_loss[0.5421] loss_occ[1.4046] 
2026-07-15 14:18:57,015 INFO [monitor.py:146] Node[0] Epoch[9] End   ==================================================
2026-07-15 14:18:57,017 INFO [monitor.py:149] Node[0] Epoch[9] Cost Time: 1462.304s
2026-07-15 14:18:57,017 INFO [metric_updater.py:360] Node[0] Epoch[9] loss_bevfusion_centerpoint_pointpillar_henet_multisensor_multitask_nuscenes_opt: car_cls_loss[0.3703] car_reg_loss[0.3640] truck_cls_loss[0.3658] truck_reg_loss[0.3586] bus_cls_loss[0.2602] bus_reg_loss[0.3375] barrier_cls_loss[0.3274] barrier_reg_loss[0.4218] bicycle_cls_loss[0.2526] bicycle_reg_loss[0.3453] pedestrian_cls_loss[0.3124] pedestrian_reg_loss[0.5420] loss_occ[1.4018] 
2026-07-15 14:18:57,017 INFO [validation.py:156] Node[0] [32mUse train `model` as val model.[0m
2026-07-15 14:18:57,839 INFO [loop_base.py:482] Node[0] [32mStart Predictor loop from epoch 9, num_epochs=1[0m
2026-07-15 14:18:57,841 INFO [fake_quantize.py:253] Node[0] Set fake quantize state to FakeQuantState.VALIDATION
2026-07-15 14:19:04,033 INFO [loop_base.py:553] Node[0] 0 / 1505
2026-07-15 14:19:31,187 INFO [loop_base.py:553] Node[0] 200 / 1505
2026-07-15 14:19:54,844 INFO [loop_base.py:553] Node[0] 400 / 1505
2026-07-15 14:20:29,356 INFO [loop_base.py:553] Node[0] 600 / 1505
2026-07-15 14:21:02,113 INFO [loop_base.py:553] Node[0] 800 / 1505
2026-07-15 14:21:33,585 INFO [loop_base.py:553] Node[0] 1000 / 1505
2026-07-15 14:22:02,117 INFO [loop_base.py:553] Node[0] 1200 / 1505
2026-07-15 14:22:27,088 INFO [loop_base.py:553] Node[0] 1400 / 1505
2026-07-15 14:22:41,046 INFO [nuscenes_metric.py:322] Node[0] The length of self.nusc_annos is: 1505
2026-07-15 14:22:43,149 INFO [nuscenes_metric.py:346] Node[0] Results writes to ./metric_results/bevfusion_centerpoint_pointpillar_henet_multisensor_multitask_nuscenes_opt/results_nusc.json
2026-07-15 14:24:36,289 INFO [nuscenes_metric.py:388] Node[0] NDS: 0.5828, mAP:0.5099
car_AP: [0.5]:0.6817  [1.0]:0.8349  [2.0]:0.8831  [4.0]:0.8972 
truck_AP: [0.5]:0.3194  [1.0]:0.5094  [2.0]:0.6221  [4.0]:0.6524 
trailer_AP: [0.5]:0.0565  [1.0]:0.2844  [2.0]:0.4029  [4.0]:0.5025 
bus_AP: [0.5]:0.3652  [1.0]:0.6161  [2.0]:0.7362  [4.0]:0.7854 
construction_vehicle_AP: [0.5]:0.0023  [1.0]:0.1087  [2.0]:0.2756  [4.0]:0.3319 
bicycle_AP: [0.5]:0.3266  [1.0]:0.4275  [2.0]:0.4438  [4.0]:0.4553 
motorcycle_AP: [0.5]:0.3879  [1.0]:0.5939  [2.0]:0.6446  [4.0]:0.6556 
pedestrian_AP: [0.5]:0.4042  [1.0]:0.5648  [2.0]:0.5957  [4.0]:0.6323 
traffic_cone_AP: [0.5]:0.4439  [1.0]:0.5708  [2.0]:0.6159  [4.0]:0.6706 
barrier_AP: [0.5]:0.2763  [1.0]:0.5495  [2.0]:0.6239  [4.0]:0.6469 

2026-07-15 14:24:36,442 WARNING [metric.py:200] Node[0] <class 'hat.metrics.nuscenes_metric.NuscenesMetric'> not ready for distributed environment, should not be used together with DistributedSampler.Might be slow in validation due to resource competition
2026-07-15 14:24:36,462 INFO [mean_iou.py:170] Node[0] ~~~~ MeanIOU Summary metrics ~~~~
Summary:
Scope                 mIoU       mAcc       aAcc
global               47.00      54.51      74.55
Per Class Results:
Class                  IoU        Acc
others                5.49       6.52
barrier              47.47      55.94
bicycle              28.29      33.79
bus                  62.64      66.30
car                  66.99      74.32
construction_vehicle      26.56      29.39
motorcycle           40.81      45.56
pedestrian           53.07      61.17
traffic_cone         28.13      32.82
trailer              44.67      60.88
truck                52.00      59.65
driveable_surface      78.21      82.66
other_flat           32.98      38.42
sidewalk             44.28      60.30
terrain              50.86      66.06
manmade              65.74      74.40
vegetation           70.78      78.44

2026-07-15 14:24:36,463 INFO [metric_updater.py:360] Node[0] Epoch[9] Validation bevfusion_centerpoint_pointpillar_henet_multisensor_multitask_nuscenes_opt: NDS[0.5828] MeanIOU[tensor(0.4700, device='cuda:0')] 
2026-07-15 14:24:42,842 INFO [checkpoint.py:233] Node[0] Save model checkpoint: ./tmp_models/bevfusion_centerpoint_pointpillar_henet_multisensor_multitask_nuscenes_opt/float-checkpoint-epoch-0009-96f7059f.pth.tar
2026-07-15 14:24:43,789 INFO [checkpoint.py:270] Node[0] Save last model checkpoint: ./tmp_models/bevfusion_centerpoint_pointpillar_henet_multisensor_multitask_nuscenes_opt/float-checkpoint-last-a1e1cfb0.pth.tar
2026-07-15 14:24:43,796 INFO [monitor.py:143] Node[0] Epoch[10] Begin ==================================================
2026-07-15 14:24:43,796 INFO [lr_updater.py:204] Node[0] Epoch[10] Step[0] GlobalStep[11730] lr=0.000252
2026-07-15 14:29:05,196 INFO [monitor.py:131] Node[0] Epoch[10] Step[0-199] Cost Time: 261.399s Speed: 4.59 samples/sec Remaining Time: 5:55:08 Remaining step percent: 57.62%
2026-07-15 14:29:06,093 INFO [metric_updater.py:360] Node[0] Epoch[10] Step[199] GlobalStep[11929] loss_bevfusion_centerpoint_pointpillar_henet_multisensor_multitask_nuscenes_opt: car_cls_loss[0.3588] car_reg_loss[0.3555] truck_cls_loss[0.3483] truck_reg_loss[0.3562] bus_cls_loss[0.2476] bus_reg_loss[0.3316] barrier_cls_loss[0.3169] barrier_reg_loss[0.4106] bicycle_cls_loss[0.2389] bicycle_reg_loss[0.3302] pedestrian_cls_loss[0.3070] pedestrian_reg_loss[0.5321] loss_occ[1.3530] 
2026-07-15 14:33:00,316 INFO [monitor.py:131] Node[0] Epoch[10] Step[200-399] Cost Time: 235.118s Speed: 5.10 samples/sec Remaining Time: 5:13:55 Remaining step percent: 56.91%
2026-07-15 14:33:01,403 INFO [metric_updater.py:360] Node[0] Epoch[10] Step[399] GlobalStep[12129] loss_bevfusion_centerpoint_pointpillar_henet_multisensor_multitask_nuscenes_opt: car_cls_loss[0.3578] car_reg_loss[0.3568] truck_cls_loss[0.3426] truck_reg_loss[0.3526] bus_cls_loss[0.2459] bus_reg_loss[0.3412] barrier_cls_loss[0.3120] barrier_reg_loss[0.4019] bicycle_cls_loss[0.2229] bicycle_reg_loss[0.3238] pedestrian_cls_loss[0.3009] pedestrian_reg_loss[0.5289] loss_occ[1.3727] 
2026-07-15 14:36:53,935 INFO [monitor.py:131] Node[0] Epoch[10] Step[400-599] Cost Time: 233.617s Speed: 5.14 samples/sec Remaining Time: 5:08:01 Remaining step percent: 56.20%
2026-07-15 14:36:54,874 INFO [metric_updater.py:360] Node[0] Epoch[10] Step[599] GlobalStep[12329] loss_bevfusion_centerpoint_pointpillar_henet_multisensor_multitask_nuscenes_opt: car_cls_loss[0.3549] car_reg_loss[0.3557] truck_cls_loss[0.3349] truck_reg_loss[0.3515] bus_cls_loss[0.2418] bus_reg_loss[0.3372] barrier_cls_loss[0.3118] barrier_reg_loss[0.4074] bicycle_cls_loss[0.2147] bicycle_reg_loss[0.3264] pedestrian_cls_loss[0.2999] pedestrian_reg_loss[0.5318] loss_occ[1.3631] 
2026-07-15 14:40:46,727 INFO [monitor.py:131] Node[0] Epoch[10] Step[600-799] Cost Time: 232.791s Speed: 5.15 samples/sec Remaining Time: 5:03:03 Remaining step percent: 55.49%
2026-07-15 14:40:47,839 INFO [metric_updater.py:360] Node[0] Epoch[10] Step[799] GlobalStep[12529] loss_bevfusion_centerpoint_pointpillar_henet_multisensor_multitask_nuscenes_opt: car_cls_loss[0.3581] car_reg_loss[0.3574] truck_cls_loss[0.3431] truck_reg_loss[0.3524] bus_cls_loss[0.2438] bus_reg_loss[0.3390] barrier_cls_loss[0.3095] barrier_reg_loss[0.4028] bicycle_cls_loss[0.2122] bicycle_reg_loss[0.3248] pedestrian_cls_loss[0.3022] pedestrian_reg_loss[0.5303] loss_occ[1.3680] 
2026-07-15 14:44:40,712 INFO [monitor.py:131] Node[0] Epoch[10] Step[800-999] Cost Time: 233.983s Speed: 5.13 samples/sec Remaining Time: 5:00:42 Remaining step percent: 54.78%
2026-07-15 14:44:41,856 INFO [metric_updater.py:360] Node[0] Epoch[10] Step[999] GlobalStep[12729] loss_bevfusion_centerpoint_pointpillar_henet_multisensor_multitask_nuscenes_opt: car_cls_loss[0.3582] car_reg_loss[0.3578] truck_cls_loss[0.3446] truck_reg_loss[0.3524] bus_cls_loss[0.2418] bus_reg_loss[0.3366] barrier_cls_loss[0.3083] barrier_reg_loss[0.4029] bicycle_cls_loss[0.2175] bicycle_reg_loss[0.3308] pedestrian_cls_loss[0.3026] pedestrian_reg_loss[0.5320] loss_occ[1.3744] 
2026-07-15 14:48:05,596 INFO [monitor.py:146] Node[0] Epoch[10] End   ==================================================
2026-07-15 14:48:05,598 INFO [monitor.py:149] Node[0] Epoch[10] Cost Time: 1401.802s
2026-07-15 14:48:05,598 INFO [metric_updater.py:360] Node[0] Epoch[10] loss_bevfusion_centerpoint_pointpillar_henet_multisensor_multitask_nuscenes_opt: car_cls_loss[0.3582] car_reg_loss[0.3575] truck_cls_loss[0.3435] truck_reg_loss[0.3518] bus_cls_loss[0.2402] bus_reg_loss[0.3339] barrier_cls_loss[0.3058] barrier_reg_loss[0.4047] bicycle_cls_loss[0.2140] bicycle_reg_loss[0.3293] pedestrian_cls_loss[0.3018] pedestrian_reg_loss[0.5318] loss_occ[1.3746] 
2026-07-15 14:48:05,599 INFO [validation.py:156] Node[0] [32mUse train `model` as val model.[0m
2026-07-15 14:48:05,767 INFO [loop_base.py:482] Node[0] [32mStart Predictor loop from epoch 10, num_epochs=1[0m
2026-07-15 14:48:05,771 INFO [fake_quantize.py:253] Node[0] Set fake quantize state to FakeQuantState.VALIDATION
2026-07-15 14:48:13,224 INFO [loop_base.py:553] Node[0] 0 / 1505
2026-07-15 14:48:43,737 INFO [loop_base.py:553] Node[0] 200 / 1505
2026-07-15 14:49:22,681 INFO [loop_base.py:553] Node[0] 400 / 1505
2026-07-15 14:49:54,605 INFO [loop_base.py:553] Node[0] 600 / 1505
2026-07-15 14:50:27,729 INFO [loop_base.py:553] Node[0] 800 / 1505
2026-07-15 14:51:01,212 INFO [loop_base.py:553] Node[0] 1000 / 1505
2026-07-15 14:51:33,492 INFO [loop_base.py:553] Node[0] 1200 / 1505
2026-07-15 14:52:09,737 INFO [loop_base.py:553] Node[0] 1400 / 1505
2026-07-15 14:52:24,152 INFO [nuscenes_metric.py:322] Node[0] The length of self.nusc_annos is: 1505
2026-07-15 14:52:26,087 INFO [nuscenes_metric.py:346] Node[0] Results writes to ./metric_results/bevfusion_centerpoint_pointpillar_henet_multisensor_multitask_nuscenes_opt/results_nusc.json
2026-07-15 14:54:30,919 INFO [nuscenes_metric.py:388] Node[0] NDS: 0.5911, mAP:0.5192
car_AP: [0.5]:0.6880  [1.0]:0.8408  [2.0]:0.8889  [4.0]:0.9033 
truck_AP: [0.5]:0.3112  [1.0]:0.5140  [2.0]:0.6157  [4.0]:0.6492 
trailer_AP: [0.5]:0.0455  [1.0]:0.2518  [2.0]:0.3790  [4.0]:0.4668 
bus_AP: [0.5]:0.3979  [1.0]:0.6505  [2.0]:0.7948  [4.0]:0.8230 
construction_vehicle_AP: [0.5]:0.0065  [1.0]:0.1233  [2.0]:0.2835  [4.0]:0.3567 
bicycle_AP: [0.5]:0.3438  [1.0]:0.4607  [2.0]:0.4740  [4.0]:0.4862 
motorcycle_AP: [0.5]:0.3956  [1.0]:0.5924  [2.0]:0.6488  [4.0]:0.6600 
pedestrian_AP: [0.5]:0.3816  [1.0]:0.5601  [2.0]:0.5946  [4.0]:0.6347 
traffic_cone_AP: [0.5]:0.4451  [1.0]:0.5789  [2.0]:0.6254  [4.0]:0.6734 
barrier_AP: [0.5]:0.3072  [1.0]:0.5804  [2.0]:0.6564  [4.0]:0.6802 

2026-07-15 14:54:31,087 WARNING [metric.py:200] Node[0] <class 'hat.metrics.nuscenes_metric.NuscenesMetric'> not ready for distributed environment, should not be used together with DistributedSampler.Might be slow in validation due to resource competition
2026-07-15 14:54:31,110 INFO [mean_iou.py:170] Node[0] ~~~~ MeanIOU Summary metrics ~~~~
Summary:
Scope                 mIoU       mAcc       aAcc
global               46.74      54.42      73.10
Per Class Results:
Class                  IoU        Acc
others                4.16       4.35
barrier              51.76      66.36
bicycle              27.92      32.49
bus                  65.05      69.56
car                  67.43      77.51
construction_vehicle      28.52      34.11
motorcycle           36.31      39.91
pedestrian           52.02      60.47
traffic_cone         28.63      34.11
trailer              45.03      52.77
truck                53.84      63.27
driveable_surface      78.09      82.71
other_flat           33.26      39.09
sidewalk             42.54      54.51
terrain              47.85      65.98
manmade              62.95      70.03
vegetation           69.19      77.86

2026-07-15 14:54:31,112 INFO [metric_updater.py:360] Node[0] Epoch[10] Validation bevfusion_centerpoint_pointpillar_henet_multisensor_multitask_nuscenes_opt: NDS[0.5911] MeanIOU[tensor(0.4674, device='cuda:0')] 
2026-07-15 14:54:38,155 INFO [checkpoint.py:233] Node[0] Save model checkpoint: ./tmp_models/bevfusion_centerpoint_pointpillar_henet_multisensor_multitask_nuscenes_opt/float-checkpoint-epoch-0010-3bf88854.pth.tar
2026-07-15 14:54:39,107 INFO [checkpoint.py:270] Node[0] Save last model checkpoint: ./tmp_models/bevfusion_centerpoint_pointpillar_henet_multisensor_multitask_nuscenes_opt/float-checkpoint-last-df695ef6.pth.tar
2026-07-15 14:54:39,115 INFO [monitor.py:143] Node[0] Epoch[11] Begin ==================================================
2026-07-15 14:54:39,116 INFO [lr_updater.py:204] Node[0] Epoch[11] Step[0] GlobalStep[12903] lr=0.000226
2026-07-15 14:59:00,228 INFO [monitor.py:131] Node[0] Epoch[11] Step[0-199] Cost Time: 261.112s Speed: 4.60 samples/sec Remaining Time: 5:29:06 Remaining step percent: 53.46%
2026-07-15 14:59:01,496 INFO [metric_updater.py:360] Node[0] Epoch[11] Step[199] GlobalStep[13102] loss_bevfusion_centerpoint_pointpillar_henet_multisensor_multitask_nuscenes_opt: car_cls_loss[0.3451] car_reg_loss[0.3526] truck_cls_loss[0.3183] truck_reg_loss[0.3481] bus_cls_loss[0.2124] bus_reg_loss[0.3099] barrier_cls_loss[0.2788] barrier_reg_loss[0.4061] bicycle_cls_loss[0.2016] bicycle_reg_loss[0.3440] pedestrian_cls_loss[0.2739] pedestrian_reg_loss[0.5259] loss_occ[1.3782] 
2026-07-15 15:02:56,508 INFO [monitor.py:131] Node[0] Epoch[11] Step[200-399] Cost Time: 236.278s Speed: 5.08 samples/sec Remaining Time: 4:52:22 Remaining step percent: 52.75%
2026-07-15 15:02:57,586 INFO [metric_updater.py:360] Node[0] Epoch[11] Step[399] GlobalStep[13302] loss_bevfusion_centerpoint_pointpillar_henet_multisensor_multitask_nuscenes_opt: car_cls_loss[0.3493] car_reg_loss[0.3518] truck_cls_loss[0.3217] truck_reg_loss[0.3432] bus_cls_loss[0.2086] bus_reg_loss[0.3164] barrier_cls_loss[0.2808] barrier_reg_loss[0.4044] bicycle_cls_loss[0.1927] bicycle_reg_loss[0.3287] pedestrian_cls_loss[0.2819] pedestrian_reg_loss[0.5236] loss_occ[1.3748] 
2026-07-15 15:06:51,152 INFO [monitor.py:131] Node[0] Epoch[11] Step[400-599] Cost Time: 234.642s Speed: 5.11 samples/sec Remaining Time: 4:46:26 Remaining step percent: 52.04%
2026-07-15 15:06:52,293 INFO [metric_updater.py:360] Node[0] Epoch[11] Step[599] GlobalStep[13502] loss_bevfusion_centerpoint_pointpillar_henet_multisensor_multitask_nuscenes_opt: car_cls_loss[0.3480] car_reg_loss[0.3519] truck_cls_loss[0.3224] truck_reg_loss[0.3424] bus_cls_loss[0.2120] bus_reg_loss[0.3194] barrier_cls_loss[0.2800] barrier_reg_loss[0.3940] bicycle_cls_loss[0.1976] bicycle_reg_loss[0.3310] pedestrian_cls_loss[0.2815] pedestrian_reg_loss[0.5262] loss_occ[1.3743] 
2026-07-15 15:10:49,546 INFO [monitor.py:131] Node[0] Epoch[11] Step[600-799] Cost Time: 238.394s Speed: 5.03 samples/sec Remaining Time: 4:47:02 Remaining step percent: 51.32%
2026-07-15 15:10:50,663 INFO [metric_updater.py:360] Node[0] Epoch[11] Step[799] GlobalStep[13702] loss_bevfusion_centerpoint_pointpillar_henet_multisensor_multitask_nuscenes_opt: car_cls_loss[0.3504] car_reg_loss[0.3517] truck_cls_loss[0.3196] truck_reg_loss[0.3418] bus_cls_loss[0.2192] bus_reg_loss[0.3186] barrier_cls_loss[0.2804] barrier_reg_loss[0.3972] bicycle_cls_loss[0.1994] bicycle_reg_loss[0.3281] pedestrian_cls_loss[0.2805] pedestrian_reg_loss[0.5251] loss_occ[1.3705] 
2026-07-15 15:14:46,481 INFO [monitor.py:131] Node[0] Epoch[11] Step[800-999] Cost Time: 236.933s Speed: 5.06 samples/sec Remaining Time: 4:41:20 Remaining step percent: 50.61%
2026-07-15 15:14:47,728 INFO [metric_updater.py:360] Node[0] Epoch[11] Step[999] GlobalStep[13902] loss_bevfusion_centerpoint_pointpillar_henet_multisensor_multitask_nuscenes_opt: car_cls_loss[0.3494] car_reg_loss[0.3515] truck_cls_loss[0.3218] truck_reg_loss[0.3426] bus_cls_loss[0.2192] bus_reg_loss[0.3181] barrier_cls_loss[0.2790] barrier_reg_loss[0.3970] bicycle_cls_loss[0.1984] bicycle_reg_loss[0.3261] pedestrian_cls_loss[0.2830] pedestrian_reg_loss[0.5252] loss_occ[1.3681] 
2026-07-15 15:18:30,868 INFO [monitor.py:146] Node[0] Epoch[11] End   ==================================================
2026-07-15 15:18:30,870 INFO [monitor.py:149] Node[0] Epoch[11] Cost Time: 1431.755s
2026-07-15 15:18:30,870 INFO [metric_updater.py:360] Node[0] Epoch[11] loss_bevfusion_centerpoint_pointpillar_henet_multisensor_multitask_nuscenes_opt: car_cls_loss[0.3496] car_reg_loss[0.3522] truck_cls_loss[0.3221] truck_reg_loss[0.3418] bus_cls_loss[0.2200] bus_reg_loss[0.3172] barrier_cls_loss[0.2842] barrier_reg_loss[0.3985] bicycle_cls_loss[0.1980] bicycle_reg_loss[0.3266] pedestrian_cls_loss[0.2852] pedestrian_reg_loss[0.5241] loss_occ[1.3693] 
2026-07-15 15:18:30,870 INFO [validation.py:156] Node[0] [32mUse train `model` as val model.[0m
2026-07-15 15:18:31,084 INFO [loop_base.py:482] Node[0] [32mStart Predictor loop from epoch 11, num_epochs=1[0m
2026-07-15 15:18:31,090 INFO [fake_quantize.py:253] Node[0] Set fake quantize state to FakeQuantState.VALIDATION
2026-07-15 15:18:36,478 INFO [loop_base.py:553] Node[0] 0 / 1505
2026-07-15 15:19:05,574 INFO [loop_base.py:553] Node[0] 200 / 1505
2026-07-15 15:19:39,996 INFO [loop_base.py:553] Node[0] 400 / 1505
2026-07-15 15:20:18,572 INFO [loop_base.py:553] Node[0] 600 / 1505
2026-07-15 15:20:56,840 INFO [loop_base.py:553] Node[0] 800 / 1505
2026-07-15 15:21:21,445 INFO [loop_base.py:553] Node[0] 1000 / 1505
2026-07-15 15:21:54,098 INFO [loop_base.py:553] Node[0] 1200 / 1505
2026-07-15 15:22:18,419 INFO [loop_base.py:553] Node[0] 1400 / 1505
2026-07-15 15:22:32,217 INFO [nuscenes_metric.py:322] Node[0] The length of self.nusc_annos is: 1505
2026-07-15 15:22:34,153 INFO [nuscenes_metric.py:346] Node[0] Results writes to ./metric_results/bevfusion_centerpoint_pointpillar_henet_multisensor_multitask_nuscenes_opt/results_nusc.json
2026-07-15 15:24:36,856 INFO [nuscenes_metric.py:388] Node[0] NDS: 0.5827, mAP:0.5070
car_AP: [0.5]:0.6870  [1.0]:0.8443  [2.0]:0.8872  [4.0]:0.9064 
truck_AP: [0.5]:0.3148  [1.0]:0.5060  [2.0]:0.5986  [4.0]:0.6394 
trailer_AP: [0.5]:0.0430  [1.0]:0.2281  [2.0]:0.3485  [4.0]:0.4996 
bus_AP: [0.5]:0.4271  [1.0]:0.6867  [2.0]:0.7959  [4.0]:0.8221 
construction_vehicle_AP: [0.5]:0.0022  [1.0]:0.1156  [2.0]:0.2648  [4.0]:0.3299 
bicycle_AP: [0.5]:0.2906  [1.0]:0.3979  [2.0]:0.4144  [4.0]:0.4245 
motorcycle_AP: [0.5]:0.3645  [1.0]:0.5523  [2.0]:0.6031  [4.0]:0.6049 
pedestrian_AP: [0.5]:0.3853  [1.0]:0.5590  [2.0]:0.5891  [4.0]:0.6255 
traffic_cone_AP: [0.5]:0.4592  [1.0]:0.5820  [2.0]:0.6210  [4.0]:0.6785 
barrier_AP: [0.5]:0.3035  [1.0]:0.5728  [2.0]:0.6396  [4.0]:0.6644 

2026-07-15 15:24:37,017 WARNING [metric.py:200] Node[0] <class 'hat.metrics.nuscenes_metric.NuscenesMetric'> not ready for distributed environment, should not be used together with DistributedSampler.Might be slow in validation due to resource competition
2026-07-15 15:24:37,041 INFO [mean_iou.py:170] Node[0] ~~~~ MeanIOU Summary metrics ~~~~
Summary:
Scope                 mIoU       mAcc       aAcc
global               47.94      55.34      76.58
Per Class Results:
Class                  IoU        Acc
others                4.06       4.21
barrier              51.23      62.33
bicycle              25.49      28.53
bus                  68.98      73.23
car                  68.18      75.42
construction_vehicle      26.53      30.82
motorcycle           33.41      35.20
pedestrian           52.07      58.83
traffic_cone         28.11      32.97
trailer              47.59      65.69
truck                55.54      65.58
driveable_surface      80.64      87.98
other_flat           35.39      41.80
sidewalk             44.51      53.40
terrain              53.59      68.33
manmade              69.17      81.08
vegetation           70.44      75.45

2026-07-15 15:24:37,043 INFO [metric_updater.py:360] Node[0] Epoch[11] Validation bevfusion_centerpoint_pointpillar_henet_multisensor_multitask_nuscenes_opt: NDS[0.5827] MeanIOU[tensor(0.4794, device='cuda:0')] 
2026-07-15 15:24:44,464 INFO [checkpoint.py:233] Node[0] Save model checkpoint: ./tmp_models/bevfusion_centerpoint_pointpillar_henet_multisensor_multitask_nuscenes_opt/float-checkpoint-epoch-0011-d9cfb913.pth.tar
2026-07-15 15:24:45,392 INFO [checkpoint.py:270] Node[0] Save last model checkpoint: ./tmp_models/bevfusion_centerpoint_pointpillar_henet_multisensor_multitask_nuscenes_opt/float-checkpoint-last-b876081d.pth.tar
2026-07-15 15:24:45,400 INFO [monitor.py:143] Node[0] Epoch[12] Begin ==================================================
2026-07-15 15:24:45,400 INFO [lr_updater.py:204] Node[0] Epoch[12] Step[0] GlobalStep[14076] lr=0.000200
2026-07-15 15:29:27,017 INFO [monitor.py:131] Node[0] Epoch[12] Step[0-199] Cost Time: 281.617s Speed: 4.26 samples/sec Remaining Time: 5:27:16 Remaining step percent: 49.29%
2026-07-15 15:29:28,163 INFO [metric_updater.py:360] Node[0] Epoch[12] Step[199] GlobalStep[14275] loss_bevfusion_centerpoint_pointpillar_henet_multisensor_multitask_nuscenes_opt: car_cls_loss[0.3419] car_reg_loss[0.3512] truck_cls_loss[0.2889] truck_reg_loss[0.3303] bus_cls_loss[0.2256] bus_reg_loss[0.3254] barrier_cls_loss[0.2730] barrier_reg_loss[0.3936] bicycle_cls_loss[0.1984] bicycle_reg_loss[0.3185] pedestrian_cls_loss[0.2911] pedestrian_reg_loss[0.5220] loss_occ[1.3449] 
2026-07-15 15:33:27,133 INFO [monitor.py:131] Node[0] Epoch[12] Step[200-399] Cost Time: 240.114s Speed: 5.00 samples/sec Remaining Time: 4:33:39 Remaining step percent: 48.58%
2026-07-15 15:33:28,283 INFO [metric_updater.py:360] Node[0] Epoch[12] Step[399] GlobalStep[14475] loss_bevfusion_centerpoint_pointpillar_henet_multisensor_multitask_nuscenes_opt: car_cls_loss[0.3416] car_reg_loss[0.3483] truck_cls_loss[0.2913] truck_reg_loss[0.3320] bus_cls_loss[0.2126] bus_reg_loss[0.3181] barrier_cls_loss[0.2775] barrier_reg_loss[0.3946] bicycle_cls_loss[0.1847] bicycle_reg_loss[0.3178] pedestrian_cls_loss[0.2822] pedestrian_reg_loss[0.5207] loss_occ[1.3307] 
2026-07-15 15:37:21,391 INFO [monitor.py:131] Node[0] Epoch[12] Step[400-599] Cost Time: 234.256s Speed: 5.12 samples/sec Remaining Time: 4:23:04 Remaining step percent: 47.87%
2026-07-15 15:37:22,553 INFO [metric_updater.py:360] Node[0] Epoch[12] Step[599] GlobalStep[14675] loss_bevfusion_centerpoint_pointpillar_henet_multisensor_multitask_nuscenes_opt: car_cls_loss[0.3400] car_reg_loss[0.3472] truck_cls_loss[0.2979] truck_reg_loss[0.3360] bus_cls_loss[0.2211] bus_reg_loss[0.3205] barrier_cls_loss[0.2831] barrier_reg_loss[0.3943] bicycle_cls_loss[0.1833] bicycle_reg_loss[0.3135] pedestrian_cls_loss[0.2800] pedestrian_reg_loss[0.5203] loss_occ[1.3283] 
2026-07-15 15:41:16,404 INFO [monitor.py:131] Node[0] Epoch[12] Step[600-799] Cost Time: 235.011s Speed: 5.11 samples/sec Remaining Time: 4:20:00 Remaining step percent: 47.16%
2026-07-15 15:41:17,379 INFO [metric_updater.py:360] Node[0] Epoch[12] Step[799] GlobalStep[14875] loss_bevfusion_centerpoint_pointpillar_henet_multisensor_multitask_nuscenes_opt: car_cls_loss[0.3421] car_reg_loss[0.3475] truck_cls_loss[0.2960] truck_reg_loss[0.3335] bus_cls_loss[0.2168] bus_reg_loss[0.3161] barrier_cls_loss[0.2782] barrier_reg_loss[0.3919] bicycle_cls_loss[0.1872] bicycle_reg_loss[0.3160] pedestrian_cls_loss[0.2776] pedestrian_reg_loss[0.5190] loss_occ[1.3317] 
2026-07-15 15:45:13,730 INFO [monitor.py:131] Node[0] Epoch[12] Step[800-999] Cost Time: 237.324s Speed: 5.06 samples/sec Remaining Time: 4:18:36 Remaining step percent: 46.45%
2026-07-15 15:45:14,821 INFO [metric_updater.py:360] Node[0] Epoch[12] Step[999] GlobalStep[15075] loss_bevfusion_centerpoint_pointpillar_henet_multisensor_multitask_nuscenes_opt: car_cls_loss[0.3415] car_reg_loss[0.3477] truck_cls_loss[0.2973] truck_reg_loss[0.3332] bus_cls_loss[0.2157] bus_reg_loss[0.3153] barrier_cls_loss[0.2811] barrier_reg_loss[0.3925] bicycle_cls_loss[0.1833] bicycle_reg_loss[0.3149] pedestrian_cls_loss[0.2793] pedestrian_reg_loss[0.5206] loss_occ[1.3342] 
2026-07-15 15:48:44,922 INFO [monitor.py:146] Node[0] Epoch[12] End   ==================================================
2026-07-15 15:48:44,924 INFO [monitor.py:149] Node[0] Epoch[12] Cost Time: 1439.524s
2026-07-15 15:48:44,924 INFO [metric_updater.py:360] Node[0] Epoch[12] loss_bevfusion_centerpoint_pointpillar_henet_multisensor_multitask_nuscenes_opt: car_cls_loss[0.3421] car_reg_loss[0.3480] truck_cls_loss[0.2968] truck_reg_loss[0.3336] bus_cls_loss[0.2110] bus_reg_loss[0.3127] barrier_cls_loss[0.2796] barrier_reg_loss[0.3923] bicycle_cls_loss[0.1849] bicycle_reg_loss[0.3153] pedestrian_cls_loss[0.2788] pedestrian_reg_loss[0.5194] loss_occ[1.3344] 
2026-07-15 15:48:44,924 INFO [validation.py:156] Node[0] [32mUse train `model` as val model.[0m
2026-07-15 15:48:45,095 INFO [loop_base.py:482] Node[0] [32mStart Predictor loop from epoch 12, num_epochs=1[0m
2026-07-15 15:48:45,098 INFO [fake_quantize.py:253] Node[0] Set fake quantize state to FakeQuantState.VALIDATION
2026-07-15 15:48:51,949 INFO [loop_base.py:553] Node[0] 0 / 1505
2026-07-15 15:49:21,024 INFO [loop_base.py:553] Node[0] 200 / 1505
2026-07-15 15:49:56,329 INFO [loop_base.py:553] Node[0] 400 / 1505
2026-07-15 15:50:25,252 INFO [loop_base.py:553] Node[0] 600 / 1505
2026-07-15 15:51:00,214 INFO [loop_base.py:553] Node[0] 800 / 1505
2026-07-15 15:51:24,603 INFO [loop_base.py:553] Node[0] 1000 / 1505
2026-07-15 15:52:01,520 INFO [loop_base.py:553] Node[0] 1200 / 1505
2026-07-15 15:52:25,872 INFO [loop_base.py:553] Node[0] 1400 / 1505
2026-07-15 15:52:39,967 INFO [nuscenes_metric.py:322] Node[0] The length of self.nusc_annos is: 1505
2026-07-15 15:52:42,030 INFO [nuscenes_metric.py:346] Node[0] Results writes to ./metric_results/bevfusion_centerpoint_pointpillar_henet_multisensor_multitask_nuscenes_opt/results_nusc.json
2026-07-15 15:54:31,889 INFO [nuscenes_metric.py:388] Node[0] NDS: 0.5912, mAP:0.5113
car_AP: [0.5]:0.6971  [1.0]:0.8462  [2.0]:0.8932  [4.0]:0.9069 
truck_AP: [0.5]:0.3204  [1.0]:0.5330  [2.0]:0.6379  [4.0]:0.6698 
trailer_AP: [0.5]:0.0760  [1.0]:0.2688  [2.0]:0.4127  [4.0]:0.4915 
bus_AP: [0.5]:0.4145  [1.0]:0.6650  [2.0]:0.7896  [4.0]:0.8162 
construction_vehicle_AP: [0.5]:0.0130  [1.0]:0.1292  [2.0]:0.2785  [4.0]:0.3467 
bicycle_AP: [0.5]:0.3109  [1.0]:0.4126  [2.0]:0.4247  [4.0]:0.4351 
motorcycle_AP: [0.5]:0.3519  [1.0]:0.5281  [2.0]:0.5789  [4.0]:0.5808 
pedestrian_AP: [0.5]:0.3899  [1.0]:0.5579  [2.0]:0.5946  [4.0]:0.6304 
traffic_cone_AP: [0.5]:0.4536  [1.0]:0.5777  [2.0]:0.6223  [4.0]:0.6694 
barrier_AP: [0.5]:0.2966  [1.0]:0.5459  [2.0]:0.6313  [4.0]:0.6543 

2026-07-15 15:54:32,035 WARNING [metric.py:200] Node[0] <class 'hat.metrics.nuscenes_metric.NuscenesMetric'> not ready for distributed environment, should not be used together with DistributedSampler.Might be slow in validation due to resource competition
2026-07-15 15:54:32,056 INFO [mean_iou.py:170] Node[0] ~~~~ MeanIOU Summary metrics ~~~~
Summary:
Scope                 mIoU       mAcc       aAcc
global               48.61      55.86      77.01
Per Class Results:
Class                  IoU        Acc
others                4.38       4.59
barrier              52.88      64.75
bicycle              25.75      28.92
bus                  67.05      71.32
car                  70.55      79.28
construction_vehicle      28.57      32.77
motorcycle           34.26      36.02
pedestrian           54.09      60.95
traffic_cone         29.87      35.38
trailer              45.89      53.28
truck                57.27      67.26
driveable_surface      80.42      88.07
other_flat           36.23      43.47
sidewalk             47.08      63.15
terrain              51.80      64.84
manmade              67.92      74.94
vegetation           72.28      80.58

2026-07-15 15:54:32,060 INFO [metric_updater.py:360] Node[0] Epoch[12] Validation bevfusion_centerpoint_pointpillar_henet_multisensor_multitask_nuscenes_opt: NDS[0.5912] MeanIOU[tensor(0.4861, device='cuda:0')] 
2026-07-15 15:54:37,490 INFO [checkpoint.py:233] Node[0] Save model checkpoint: ./tmp_models/bevfusion_centerpoint_pointpillar_henet_multisensor_multitask_nuscenes_opt/float-checkpoint-epoch-0012-a17b7258.pth.tar
2026-07-15 15:54:38,284 INFO [checkpoint.py:270] Node[0] Save last model checkpoint: ./tmp_models/bevfusion_centerpoint_pointpillar_henet_multisensor_multitask_nuscenes_opt/float-checkpoint-last-f31d43c6.pth.tar
2026-07-15 15:54:38,292 INFO [monitor.py:143] Node[0] Epoch[13] Begin ==================================================
2026-07-15 15:54:38,292 INFO [lr_updater.py:204] Node[0] Epoch[13] Step[0] GlobalStep[15249] lr=0.000174
2026-07-15 15:58:59,571 INFO [monitor.py:131] Node[0] Epoch[13] Step[0-199] Cost Time: 261.279s Speed: 4.59 samples/sec Remaining Time: 4:37:58 Remaining step percent: 45.12%
2026-07-15 15:59:00,617 INFO [metric_updater.py:360] Node[0] Epoch[13] Step[199] GlobalStep[15448] loss_bevfusion_centerpoint_pointpillar_henet_multisensor_multitask_nuscenes_opt: car_cls_loss[0.3281] car_reg_loss[0.3392] truck_cls_loss[0.2927] truck_reg_loss[0.3313] bus_cls_loss[0.1866] bus_reg_loss[0.3142] barrier_cls_loss[0.2244] barrier_reg_loss[0.3761] bicycle_cls_loss[0.1506] bicycle_reg_loss[0.3141] pedestrian_cls_loss[0.2642] pedestrian_reg_loss[0.5099] loss_occ[1.3285] 
2026-07-15 16:02:57,608 INFO [monitor.py:131] Node[0] Epoch[13] Step[200-399] Cost Time: 238.036s Speed: 5.04 samples/sec Remaining Time: 4:08:00 Remaining step percent: 44.41%
2026-07-15 16:02:58,780 INFO [metric_updater.py:360] Node[0] Epoch[13] Step[399] GlobalStep[15648] loss_bevfusion_centerpoint_pointpillar_henet_multisensor_multitask_nuscenes_opt: car_cls_loss[0.3309] car_reg_loss[0.3409] truck_cls_loss[0.2827] truck_reg_loss[0.3293] bus_cls_loss[0.1818] bus_reg_loss[0.3093] barrier_cls_loss[0.2442] barrier_reg_loss[0.3881] bicycle_cls_loss[0.1604] bicycle_reg_loss[0.3118] pedestrian_cls_loss[0.2680] pedestrian_reg_loss[0.5127] loss_occ[1.3294] 
2026-07-15 16:07:01,662 INFO [monitor.py:131] Node[0] Epoch[13] Step[400-599] Cost Time: 244.050s Speed: 4.92 samples/sec Remaining Time: 4:10:12 Remaining step percent: 43.70%
2026-07-15 16:07:03,639 INFO [metric_updater.py:360] Node[0] Epoch[13] Step[599] GlobalStep[15848] loss_bevfusion_centerpoint_pointpillar_henet_multisensor_multitask_nuscenes_opt: car_cls_loss[0.3343] car_reg_loss[0.3435] truck_cls_loss[0.2799] truck_reg_loss[0.3291] bus_cls_loss[0.1863] bus_reg_loss[0.3092] barrier_cls_loss[0.2455] barrier_reg_loss[0.3878] bicycle_cls_loss[0.1657] bicycle_reg_loss[0.3130] pedestrian_cls_loss[0.2650] pedestrian_reg_loss[0.5122] loss_occ[1.3291] 
2026-07-15 16:11:18,790 INFO [monitor.py:131] Node[0] Epoch[13] Step[600-799] Cost Time: 257.121s Speed: 4.67 samples/sec Remaining Time: 4:19:19 Remaining step percent: 42.99%
2026-07-15 16:11:19,967 INFO [metric_updater.py:360] Node[0] Epoch[13] Step[799] GlobalStep[16048] loss_bevfusion_centerpoint_pointpillar_henet_multisensor_multitask_nuscenes_opt: car_cls_loss[0.3362] car_reg_loss[0.3446] truck_cls_loss[0.2802] truck_reg_loss[0.3285] bus_cls_loss[0.1867] bus_reg_loss[0.3074] barrier_cls_loss[0.2502] barrier_reg_loss[0.3884] bicycle_cls_loss[0.1657] bicycle_reg_loss[0.3113] pedestrian_cls_loss[0.2646] pedestrian_reg_loss[0.5125] loss_occ[1.3282] 
2026-07-15 16:15:21,848 INFO [monitor.py:131] Node[0] Epoch[13] Step[800-999] Cost Time: 243.056s Speed: 4.94 samples/sec Remaining Time: 4:01:05 Remaining step percent: 42.28%
2026-07-15 16:15:23,073 INFO [metric_updater.py:360] Node[0] Epoch[13] Step[999] GlobalStep[16248] loss_bevfusion_centerpoint_pointpillar_henet_multisensor_multitask_nuscenes_opt: car_cls_loss[0.3376] car_reg_loss[0.3451] truck_cls_loss[0.2779] truck_reg_loss[0.3263] bus_cls_loss[0.1926] bus_reg_loss[0.3040] barrier_cls_loss[0.2547] barrier_reg_loss[0.3859] bicycle_cls_loss[0.1695] bicycle_reg_loss[0.3152] pedestrian_cls_loss[0.2662] pedestrian_reg_loss[0.5134] loss_occ[1.3317] 
2026-07-15 16:18:48,744 INFO [monitor.py:146] Node[0] Epoch[13] End   ==================================================
2026-07-15 16:18:48,745 INFO [monitor.py:149] Node[0] Epoch[13] Cost Time: 1450.454s
2026-07-15 16:18:48,745 INFO [metric_updater.py:360] Node[0] Epoch[13] loss_bevfusion_centerpoint_pointpillar_henet_multisensor_multitask_nuscenes_opt: car_cls_loss[0.3382] car_reg_loss[0.3457] truck_cls_loss[0.2751] truck_reg_loss[0.3252] bus_cls_loss[0.1911] bus_reg_loss[0.3012] barrier_cls_loss[0.2567] barrier_reg_loss[0.3859] bicycle_cls_loss[0.1685] bicycle_reg_loss[0.3170] pedestrian_cls_loss[0.2675] pedestrian_reg_loss[0.5133] loss_occ[1.3313] 
2026-07-15 16:18:48,746 INFO [validation.py:156] Node[0] [32mUse train `model` as val model.[0m
2026-07-15 16:18:49,414 INFO [loop_base.py:482] Node[0] [32mStart Predictor loop from epoch 13, num_epochs=1[0m
2026-07-15 16:18:49,416 INFO [fake_quantize.py:253] Node[0] Set fake quantize state to FakeQuantState.VALIDATION
2026-07-15 16:18:57,067 INFO [loop_base.py:553] Node[0] 0 / 1505
2026-07-15 16:19:27,836 INFO [loop_base.py:553] Node[0] 200 / 1505
2026-07-15 16:20:01,680 INFO [loop_base.py:553] Node[0] 400 / 1505
2026-07-15 16:20:42,948 INFO [loop_base.py:553] Node[0] 600 / 1505
2026-07-15 16:21:18,108 INFO [loop_base.py:553] Node[0] 800 / 1505
2026-07-15 16:21:55,854 INFO [loop_base.py:553] Node[0] 1000 / 1505
2026-07-15 16:22:34,162 INFO [loop_base.py:553] Node[0] 1200 / 1505
2026-07-15 16:23:00,644 INFO [loop_base.py:553] Node[0] 1400 / 1505
2026-07-15 16:23:14,666 INFO [nuscenes_metric.py:322] Node[0] The length of self.nusc_annos is: 1505
2026-07-15 16:23:16,166 INFO [nuscenes_metric.py:346] Node[0] Results writes to ./metric_results/bevfusion_centerpoint_pointpillar_henet_multisensor_multitask_nuscenes_opt/results_nusc.json
2026-07-15 16:25:19,723 INFO [nuscenes_metric.py:388] Node[0] NDS: 0.5858, mAP:0.5002
car_AP: [0.5]:0.6880  [1.0]:0.8424  [2.0]:0.8849  [4.0]:0.9034 
truck_AP: [0.5]:0.2879  [1.0]:0.4807  [2.0]:0.5901  [4.0]:0.6160 
trailer_AP: [0.5]:0.0553  [1.0]:0.2537  [2.0]:0.3986  [4.0]:0.5172 
bus_AP: [0.5]:0.3929  [1.0]:0.6433  [2.0]:0.7689  [4.0]:0.7971 
construction_vehicle_AP: [0.5]:0.0114  [1.0]:0.1361  [2.0]:0.2617  [4.0]:0.3330 
bicycle_AP: [0.5]:0.2759  [1.0]:0.3684  [2.0]:0.3845  [4.0]:0.3889 
motorcycle_AP: [0.5]:0.3533  [1.0]:0.5473  [2.0]:0.5825  [4.0]:0.5932 
pedestrian_AP: [0.5]:0.3857  [1.0]:0.5571  [2.0]:0.5862  [4.0]:0.6209 
traffic_cone_AP: [0.5]:0.4640  [1.0]:0.5842  [2.0]:0.6259  [4.0]:0.6806 
barrier_AP: [0.5]:0.2960  [1.0]:0.5585  [2.0]:0.6337  [4.0]:0.6570 

2026-07-15 16:25:19,879 WARNING [metric.py:200] Node[0] <class 'hat.metrics.nuscenes_metric.NuscenesMetric'> not ready for distributed environment, should not be used together with DistributedSampler.Might be slow in validation due to resource competition
2026-07-15 16:25:19,895 INFO [mean_iou.py:170] Node[0] ~~~~ MeanIOU Summary metrics ~~~~
Summary:
Scope                 mIoU       mAcc       aAcc
global               48.44      55.35      76.97
Per Class Results:
Class                  IoU        Acc
others                4.60       4.82
barrier              51.08      59.98
bicycle              24.19      26.32
bus                  67.29      72.21
car                  69.98      78.28
construction_vehicle      29.90      35.86
motorcycle           35.22      36.87
pedestrian           53.16      59.14
traffic_cone         31.30      36.88
trailer              48.16      60.76
truck                53.78      61.06
driveable_surface      80.86      88.07
other_flat           33.23      37.12
sidewalk             46.42      57.47
terrain              53.90      69.62
manmade              69.83      80.69
vegetation           70.54      75.77

2026-07-15 16:25:19,896 INFO [metric_updater.py:360] Node[0] Epoch[13] Validation bevfusion_centerpoint_pointpillar_henet_multisensor_multitask_nuscenes_opt: NDS[0.5858] MeanIOU[tensor(0.4844, device='cuda:0')] 
2026-07-15 16:25:26,563 INFO [checkpoint.py:233] Node[0] Save model checkpoint: ./tmp_models/bevfusion_centerpoint_pointpillar_henet_multisensor_multitask_nuscenes_opt/float-checkpoint-epoch-0013-d18c6f06.pth.tar
2026-07-15 16:25:27,447 INFO [checkpoint.py:270] Node[0] Save last model checkpoint: ./tmp_models/bevfusion_centerpoint_pointpillar_henet_multisensor_multitask_nuscenes_opt/float-checkpoint-last-e2068ea6.pth.tar
2026-07-15 16:25:27,455 INFO [monitor.py:143] Node[0] Epoch[14] Begin ==================================================
2026-07-15 16:25:27,455 INFO [lr_updater.py:204] Node[0] Epoch[14] Step[0] GlobalStep[16422] lr=0.000148
2026-07-15 16:29:45,624 INFO [monitor.py:131] Node[0] Epoch[14] Step[0-199] Cost Time: 258.166s Speed: 4.65 samples/sec Remaining Time: 4:09:18 Remaining step percent: 40.96%
2026-07-15 16:29:46,695 INFO [metric_updater.py:360] Node[0] Epoch[14] Step[199] GlobalStep[16621] loss_bevfusion_centerpoint_pointpillar_henet_multisensor_multitask_nuscenes_opt: car_cls_loss[0.3255] car_reg_loss[0.3390] truck_cls_loss[0.2528] truck_reg_loss[0.3164] bus_cls_loss[0.1639] bus_reg_loss[0.2735] barrier_cls_loss[0.2432] barrier_reg_loss[0.3750] bicycle_cls_loss[0.1417] bicycle_reg_loss[0.2839] pedestrian_cls_loss[0.2531] pedestrian_reg_loss[0.5056] loss_occ[1.3000] 
2026-07-15 16:33:43,173 INFO [monitor.py:131] Node[0] Epoch[14] Step[200-399] Cost Time: 237.547s Speed: 5.05 samples/sec Remaining Time: 3:44:17 Remaining step percent: 40.25%
2026-07-15 16:33:44,330 INFO [metric_updater.py:360] Node[0] Epoch[14] Step[399] GlobalStep[16821] loss_bevfusion_centerpoint_pointpillar_henet_multisensor_multitask_nuscenes_opt: car_cls_loss[0.3263] car_reg_loss[0.3389] truck_cls_loss[0.2484] truck_reg_loss[0.3187] bus_cls_loss[0.1681] bus_reg_loss[0.2839] barrier_cls_loss[0.2360] barrier_reg_loss[0.3691] bicycle_cls_loss[0.1610] bicycle_reg_loss[0.2992] pedestrian_cls_loss[0.2605] pedestrian_reg_loss[0.5120] loss_occ[1.3030] 
2026-07-15 16:37:48,732 INFO [monitor.py:131] Node[0] Epoch[14] Step[400-599] Cost Time: 245.557s Speed: 4.89 samples/sec Remaining Time: 3:47:45 Remaining step percent: 39.54%
2026-07-15 16:37:49,813 INFO [metric_updater.py:360] Node[0] Epoch[14] Step[599] GlobalStep[17021] loss_bevfusion_centerpoint_pointpillar_henet_multisensor_multitask_nuscenes_opt: car_cls_loss[0.3233] car_reg_loss[0.3406] truck_cls_loss[0.2527] truck_reg_loss[0.3207] bus_cls_loss[0.1728] bus_reg_loss[0.2872] barrier_cls_loss[0.2389] barrier_reg_loss[0.3709] bicycle_cls_loss[0.1531] bicycle_reg_loss[0.2973] pedestrian_cls_loss[0.2589] pedestrian_reg_loss[0.5099] loss_occ[1.3110] 
2026-07-15 16:41:49,607 INFO [monitor.py:131] Node[0] Epoch[14] Step[600-799] Cost Time: 240.873s Speed: 4.98 samples/sec Remaining Time: 3:39:23 Remaining step percent: 38.82%
2026-07-15 16:41:50,606 INFO [metric_updater.py:360] Node[0] Epoch[14] Step[799] GlobalStep[17221] loss_bevfusion_centerpoint_pointpillar_henet_multisensor_multitask_nuscenes_opt: car_cls_loss[0.3240] car_reg_loss[0.3420] truck_cls_loss[0.2551] truck_reg_loss[0.3220] bus_cls_loss[0.1749] bus_reg_loss[0.2866] barrier_cls_loss[0.2444] barrier_reg_loss[0.3741] bicycle_cls_loss[0.1529] bicycle_reg_loss[0.2978] pedestrian_cls_loss[0.2588] pedestrian_reg_loss[0.5077] loss_occ[1.3143] 
2026-07-15 16:45:45,983 INFO [monitor.py:131] Node[0] Epoch[14] Step[800-999] Cost Time: 236.375s Speed: 5.08 samples/sec Remaining Time: 3:31:21 Remaining step percent: 38.11%
2026-07-15 16:45:47,127 INFO [metric_updater.py:360] Node[0] Epoch[14] Step[999] GlobalStep[17421] loss_bevfusion_centerpoint_pointpillar_henet_multisensor_multitask_nuscenes_opt: car_cls_loss[0.3253] car_reg_loss[0.3418] truck_cls_loss[0.2548] truck_reg_loss[0.3222] bus_cls_loss[0.1756] bus_reg_loss[0.2887] barrier_cls_loss[0.2480] barrier_reg_loss[0.3729] bicycle_cls_loss[0.1554] bicycle_reg_loss[0.3004] pedestrian_cls_loss[0.2571] pedestrian_reg_loss[0.5075] loss_occ[1.3096] 
2026-07-15 16:49:15,465 INFO [monitor.py:146] Node[0] Epoch[14] End   ==================================================
2026-07-15 16:49:15,467 INFO [monitor.py:149] Node[0] Epoch[14] Cost Time: 1428.012s
2026-07-15 16:49:15,467 INFO [metric_updater.py:360] Node[0] Epoch[14] loss_bevfusion_centerpoint_pointpillar_henet_multisensor_multitask_nuscenes_opt: car_cls_loss[0.3240] car_reg_loss[0.3411] truck_cls_loss[0.2523] truck_reg_loss[0.3210] bus_cls_loss[0.1763] bus_reg_loss[0.2891] barrier_cls_loss[0.2509] barrier_reg_loss[0.3736] bicycle_cls_loss[0.1561] bicycle_reg_loss[0.2980] pedestrian_cls_loss[0.2579] pedestrian_reg_loss[0.5066] loss_occ[1.3081] 
2026-07-15 16:49:15,468 INFO [validation.py:156] Node[0] [32mUse train `model` as val model.[0m
2026-07-15 16:49:15,636 INFO [loop_base.py:482] Node[0] [32mStart Predictor loop from epoch 14, num_epochs=1[0m
2026-07-15 16:49:15,639 INFO [fake_quantize.py:253] Node[0] Set fake quantize state to FakeQuantState.VALIDATION
2026-07-15 16:49:21,214 INFO [loop_base.py:553] Node[0] 0 / 1505
2026-07-15 16:50:03,080 INFO [loop_base.py:553] Node[0] 200 / 1505
2026-07-15 16:50:46,492 INFO [loop_base.py:553] Node[0] 400 / 1505
2026-07-15 16:51:24,600 INFO [loop_base.py:553] Node[0] 600 / 1505
2026-07-15 16:51:54,180 INFO [loop_base.py:553] Node[0] 800 / 1505
2026-07-15 16:52:33,021 INFO [loop_base.py:553] Node[0] 1000 / 1505
2026-07-15 16:53:00,494 INFO [loop_base.py:553] Node[0] 1200 / 1505
2026-07-15 16:53:27,526 INFO [loop_base.py:553] Node[0] 1400 / 1505
2026-07-15 16:53:43,427 INFO [nuscenes_metric.py:322] Node[0] The length of self.nusc_annos is: 1505
2026-07-15 16:53:45,678 INFO [nuscenes_metric.py:346] Node[0] Results writes to ./metric_results/bevfusion_centerpoint_pointpillar_henet_multisensor_multitask_nuscenes_opt/results_nusc.json
2026-07-15 16:55:46,956 INFO [nuscenes_metric.py:388] Node[0] NDS: 0.5840, mAP:0.4954
car_AP: [0.5]:0.6878  [1.0]:0.8356  [2.0]:0.8836  [4.0]:0.8972 
truck_AP: [0.5]:0.3039  [1.0]:0.5022  [2.0]:0.5926  [4.0]:0.6340 
trailer_AP: [0.5]:0.0753  [1.0]:0.2504  [2.0]:0.3695  [4.0]:0.4543 
bus_AP: [0.5]:0.4245  [1.0]:0.6474  [2.0]:0.7644  [4.0]:0.7954 
construction_vehicle_AP: [0.5]:0.0094  [1.0]:0.1466  [2.0]:0.3019  [4.0]:0.3668 
bicycle_AP: [0.5]:0.2640  [1.0]:0.3426  [2.0]:0.3508  [4.0]:0.3548 
motorcycle_AP: [0.5]:0.3422  [1.0]:0.4959  [2.0]:0.5432  [4.0]:0.5451 
pedestrian_AP: [0.5]:0.3797  [1.0]:0.5519  [2.0]:0.5872  [4.0]:0.6183 
traffic_cone_AP: [0.5]:0.4411  [1.0]:0.5700  [2.0]:0.6137  [4.0]:0.6632 
barrier_AP: [0.5]:0.3134  [1.0]:0.5791  [2.0]:0.6458  [4.0]:0.6694 

2026-07-15 16:55:47,174 WARNING [metric.py:200] Node[0] <class 'hat.metrics.nuscenes_metric.NuscenesMetric'> not ready for distributed environment, should not be used together with DistributedSampler.Might be slow in validation due to resource competition
2026-07-15 16:55:47,193 INFO [mean_iou.py:170] Node[0] ~~~~ MeanIOU Summary metrics ~~~~
Summary:
Scope                 mIoU       mAcc       aAcc
global               48.80      57.28      76.47
Per Class Results:
Class                  IoU        Acc
others                4.11       4.20
barrier              54.22      67.88
bicycle              25.30      27.34
bus                  66.73      72.75
car                  71.22      80.34
construction_vehicle      31.47      37.58
motorcycle           34.46      36.02
pedestrian           55.61      62.91
traffic_cone         32.32      39.24
trailer              44.18      63.22
truck                56.72      65.87
driveable_surface      79.65      85.62
other_flat           34.20      47.96
sidewalk             46.67      60.90
terrain              52.86      66.82
manmade              67.63      74.77
vegetation           72.29      80.37

2026-07-15 16:55:47,195 INFO [metric_updater.py:360] Node[0] Epoch[14] Validation bevfusion_centerpoint_pointpillar_henet_multisensor_multitask_nuscenes_opt: NDS[0.5840] MeanIOU[tensor(0.4880, device='cuda:0')] 
2026-07-15 16:55:55,131 INFO [checkpoint.py:233] Node[0] Save model checkpoint: ./tmp_models/bevfusion_centerpoint_pointpillar_henet_multisensor_multitask_nuscenes_opt/float-checkpoint-epoch-0014-535a2d17.pth.tar
2026-07-15 16:55:56,096 INFO [checkpoint.py:270] Node[0] Save last model checkpoint: ./tmp_models/bevfusion_centerpoint_pointpillar_henet_multisensor_multitask_nuscenes_opt/float-checkpoint-last-32de4add.pth.tar
2026-07-15 16:55:56,104 INFO [monitor.py:143] Node[0] Epoch[15] Begin ==================================================
2026-07-15 16:55:56,104 INFO [lr_updater.py:204] Node[0] Epoch[15] Step[0] GlobalStep[17595] lr=0.000124
2026-07-15 17:00:27,895 INFO [monitor.py:131] Node[0] Epoch[15] Step[0-199] Cost Time: 271.788s Speed: 4.42 samples/sec Remaining Time: 3:55:45 Remaining step percent: 36.79%
2026-07-15 17:00:29,061 INFO [metric_updater.py:360] Node[0] Epoch[15] Step[199] GlobalStep[17794] loss_bevfusion_centerpoint_pointpillar_henet_multisensor_multitask_nuscenes_opt: car_cls_loss[0.3190] car_reg_loss[0.3395] truck_cls_loss[0.2523] truck_reg_loss[0.3211] bus_cls_loss[0.1609] bus_reg_loss[0.2819] barrier_cls_loss[0.2518] barrier_reg_loss[0.3913] bicycle_cls_loss[0.1482] bicycle_reg_loss[0.2872] pedestrian_cls_loss[0.2546] pedestrian_reg_loss[0.5078] loss_occ[1.3165] 
2026-07-15 17:04:34,104 INFO [monitor.py:131] Node[0] Epoch[15] Step[200-399] Cost Time: 246.207s Speed: 4.87 samples/sec Remaining Time: 3:28:23 Remaining step percent: 36.08%
2026-07-15 17:04:35,256 INFO [metric_updater.py:360] Node[0] Epoch[15] Step[399] GlobalStep[17994] loss_bevfusion_centerpoint_pointpillar_henet_multisensor_multitask_nuscenes_opt: car_cls_loss[0.3240] car_reg_loss[0.3411] truck_cls_loss[0.2482] truck_reg_loss[0.3149] bus_cls_loss[0.1678] bus_reg_loss[0.2839] barrier_cls_loss[0.2504] barrier_reg_loss[0.3819] bicycle_cls_loss[0.1458] bicycle_reg_loss[0.2922] pedestrian_cls_loss[0.2527] pedestrian_reg_loss[0.4985] loss_occ[1.3012] 
