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