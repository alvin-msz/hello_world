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