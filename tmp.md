mAP: 0.0000
mATE: 1.0000
mASE: 1.0000
mAOE: 1.0000
mAVE: 1.0000
mAAE: 1.0000
NDS: 0.0000
Eval time: 195.3s

Per-class results:
Object Class	AP	ATE	ASE	AOE	AVE	AAE
car	0.000	1.000	1.000	1.000	1.000	1.000
truck	0.000	1.000	1.000	1.000	1.000	1.000
bus	0.000	1.000	1.000	1.000	1.000	1.000
trailer	0.000	1.000	1.000	1.000	1.000	1.000
construction_vehicle	0.000	1.000	1.000	1.000	1.000	1.000
pedestrian	0.000	1.000	1.000	1.000	1.000	1.000
motorcycle	0.000	1.000	1.000	1.000	1.000	1.000
bicycle	0.000	1.000	1.000	1.000	1.000	1.000
traffic_cone	0.000	1.000	1.000	nan	nan	nan
barrier	0.000	1.000	1.000	1.000	nan	nan
2026-07-06 11:29:57,559 INFO [nuscenes_metric.py:388] Node[0] NDS: 0.0000, mAP:0.0000
car_AP: [0.5]:0.0000  [1.0]:0.0000  [2.0]:0.0000  [4.0]:0.0000 
truck_AP: [0.5]:0.0000  [1.0]:0.0000  [2.0]:0.0000  [4.0]:0.0000 
trailer_AP: [0.5]:0.0000  [1.0]:0.0000  [2.0]:0.0000  [4.0]:0.0000 
bus_AP: [0.5]:0.0000  [1.0]:0.0000  [2.0]:0.0000  [4.0]:0.0000 
construction_vehicle_AP: [0.5]:0.0000  [1.0]:0.0000  [2.0]:0.0000  [4.0]:0.0000 
bicycle_AP: [0.5]:0.0000  [1.0]:0.0000  [2.0]:0.0000  [4.0]:0.0000 
motorcycle_AP: [0.5]:0.0000  [1.0]:0.0000  [2.0]:0.0000  [4.0]:0.0000 
pedestrian_AP: [0.5]:0.0000  [1.0]:0.0000  [2.0]:0.0000  [4.0]:0.0000 
traffic_cone_AP: [0.5]:0.0000  [1.0]:0.0000  [2.0]:0.0000  [4.0]:0.0000 
barrier_AP: [0.5]:0.0000  [1.0]:0.0000  [2.0]:0.0000  [4.0]:0.0000 

2026-07-06 11:29:58,245 WARNING [metric.py:200] Node[0] <class 'hat.metrics.nuscenes_metric.NuscenesMetric'> not ready for distributed environment, should not be used together with DistributedSampler.Might be slow in validation due to resource competition
2026-07-06 11:29:58,327 INFO [mean_iou.py:170] Node[0] ~~~~ MeanIOU Summary metrics ~~~~
Summary:
Scope                 mIoU       mAcc       aAcc
global               45.91      53.64      70.00
Per Class Results:
Class                  IoU        Acc
others                3.59       3.70
barrier              49.70      61.90
bicycle              25.63      31.36
bus                  61.56      64.97
car                  67.86      80.54
construction_vehicle      22.07      23.05
motorcycle           38.07      45.52
pedestrian           54.43      59.86
traffic_cone         34.75      48.07
trailer              41.12      52.32
truck                54.55      64.38
driveable_surface      79.58      88.72
other_flat           30.86      35.96
sidewalk             43.41      53.76
terrain              50.69      64.61
manmade              60.83      67.45
vegetation           61.69      65.77