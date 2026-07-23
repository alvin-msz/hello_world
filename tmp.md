input[0]: 
name: _input_0
valid shape: (1,6,3,256,704)
aligned byte size: 6488064
tensor type: HB_DNN_TENSOR_TYPE_S16
quanti type: SCALE
stride: (6488064,1081344,360448,1408,2)
scale data: (0.0177)
zero_point data: (0)

input[1]: 
name: _input_1
valid shape: (1,6,160000,2)
aligned byte size: 3840000
tensor type: HB_DNN_TENSOR_TYPE_S16
quanti type: SCALE
stride: (3840000,640000,4,2)
scale data: (0.00323519)
zero_point data: (0)

input[2]: 
name: _input_2
valid shape: (1,6,160000)
aligned byte size: 1920000
tensor type: HB_DNN_TENSOR_TYPE_S16
quanti type: SCALE
stride: (1920000,320000,2)
scale data: (0.00137362)
zero_point data: (0)

input[3]: 
name: _input_3
valid shape: (1,6,160000)
aligned byte size: 960000
tensor type: HB_DNN_TENSOR_TYPE_BOOL8
quanti type: NONE
stride: (960000,160000,1)

input[4]: 
name: _input_4
valid shape: (35000,4)
aligned byte size: 560000
tensor type: HB_DNN_TENSOR_TYPE_F32
quanti type: NONE
stride: (16,4)

input[5]: 
name: _input_5
valid shape: (1,384,200,200)
aligned byte size: 19660800
tensor type: HB_DNN_TENSOR_TYPE_S8
quanti type: SCALE
stride: (19660800,51200,256,1)
scale data: (0.0472441)
zero_point data: (0)

input[6]: 
name: _input_6
valid shape: (1,4)
aligned byte size: 128
tensor type: HB_DNN_TENSOR_TYPE_F32
quanti type: NONE
stride: (16,4)

output[0]: 
name: _output_0
valid shape: (1,200,200,16)
aligned byte size: 2560000
tensor type: HB_DNN_TENSOR_TYPE_S32
quanti type: NONE
stride: (2560000,12800,64,4)

output[1]: 
name: _output_1
valid shape: (1,384,200,200)
aligned byte size: 19660800
tensor type: HB_DNN_TENSOR_TYPE_S8
quanti type: SCALE
stride: (19660800,51200,256,1)
scale data: (0.0463431)
zero_point data: (0)

---------------------------------------------------------------------



ego_motion.bin  imgs.bin  points.bin  uv.bin  valid.bin  zcam.bin


I20260318 16:37:15.461673 520055 qat_bevfusion_occ_4d_post_process_method.cc:295] BevFusion OCC latency frame_id=0 pre_ms=0.732 infer_ms=101.913 post_ms=0.418
I20260318 16:37:15.617733 520055 qat_bevfusion_occ_4d_post_process_method.cc:295] BevFusion OCC latency frame_id=1 pre_ms=61.110 infer_ms=100.171 post_ms=0.301
I20260318 16:37:15.723694 520055 qat_bevfusion_occ_4d_post_process_method.cc:295] BevFusion OCC latency frame_id=2 pre_ms=60.554 infer_ms=100.391 post_ms=0.268
I20260318 16:37:15.829391 520055 qat_bevfusion_occ_4d_post_process_method.cc:295] BevFusion OCC latency frame_id=3 pre_ms=59.479 infer_ms=100.061 post_ms=0.269
I20260318 16:37:15.934947 520055 qat_bevfusion_occ_4d_post_process_method.cc:295] BevFusion OCC latency frame_id=4 pre_ms=59.218 infer_ms=100.037 post_ms=0.268
I20260318 16:37:16.040242 520055 qat_bevfusion_occ_4d_post_process_method.cc:295] BevFusion OCC latency frame_id=5 pre_ms=59.188 infer_ms=99.802 post_ms=0.255
I20260318 16:37:16.146133 520055 qat_bevfusion_occ_4d_post_process_method.cc:295] BevFusion OCC latency frame_id=6 pre_ms=59.028 infer_ms=100.203 post_ms=0.264
I20260318 16:37:16.251732 520055 qat_bevfusion_occ_4d_post_process_method.cc:295] BevFusion OCC latency frame_id=7 pre_ms=58.954 infer_ms=100.022 post_ms=0.259
I20260318 16:37:16.357484 520055 qat_bevfusion_occ_4d_post_process_method.cc:295] BevFusion OCC latency frame_id=8 pre_ms=59.150 infer_ms=100.220 post_ms=0.278
I20260318 16:37:16.462821 520055 qat_bevfusion_occ_4d_post_process_method.cc:295] BevFusion OCC latency frame_id=9 pre_ms=60.649 infer_ms=99.760 post_ms=0.275
I20260318 16:37:16.568475 520055 qat_bevfusion_occ_4d_post_process_method.cc:295] BevFusion OCC latency frame_id=10 pre_ms=59.096 infer_ms=100.064 post_ms=0.254
I20260318 16:37:16.674043 520055 qat_bevfusion_occ_4d_post_process_method.cc:295] BevFusion OCC latency frame_id=11 pre_ms=59.126 infer_ms=99.761 post_ms=0.276
I20260318 16:37:16.779805 520055 qat_bevfusion_occ_4d_post_process_method.cc:295] BevFusion OCC latency frame_id=12 pre_ms=59.027 infer_ms=100.079 post_ms=0.275
I20260318 16:37:16.885306 520055 qat_bevfusion_occ_4d_post_process_method.cc:295] BevFusion OCC latency frame_id=13 pre_ms=59.069 infer_ms=99.828 post_ms=0.277
I20260318 16:37:16.991029 520055 qat_bevfusion_occ_4d_post_process_method.cc:295] BevFusion OCC latency frame_id=14 pre_ms=59.163 infer_ms=100.044 post_ms=0.275
I20260318 16:37:17.096432 520055 qat_bevfusion_occ_4d_post_process_method.cc:295] BevFusion OCC latency frame_id=15 pre_ms=58.994 infer_ms=99.816 post_ms=0.275
I20260318 16:37:17.202836 520055 qat_bevfusion_occ_4d_post_process_method.cc:295] BevFusion OCC latency frame_id=16 pre_ms=59.150 infer_ms=99.950 post_ms=0.275