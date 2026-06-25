input[0]: 
name: features
valid shape: (1,5,20,30000)
aligned byte size: 3008000
tensor type: HB_DNN_TENSOR_TYPE_S8
quanti type: SCALE
stride: (3008000,601600,30080,1)
scale data: (0.00771821)
zero_point data: (0)

input[1]: 
name: coors
valid shape: (30000,4)
aligned byte size: 480000
tensor type: HB_DNN_TENSOR_TYPE_S32
quanti type: NONE
stride: (16,4)

input[2]: 
name: img
valid shape: (6,3,512,960)
aligned byte size: 9437184
tensor type: HB_DNN_TENSOR_TYPE_S8
quanti type: SCALE
stride: (1572864,524288,1024,1)
scale data: (0.00686275)
zero_point data: (0)

input[3]: 
name: ego2img
valid shape: (6,4,4)
aligned byte size: 384
tensor type: HB_DNN_TENSOR_TYPE_F32
quanti type: NONE
stride: (64,16,4)

input[4]: 
name: points0
valid shape: (10,128,128,2)
aligned byte size: 655360
tensor type: HB_DNN_TENSOR_TYPE_S16
quanti type: SCALE
stride: (65536,512,4,2)
scale data: (0.0078125)
zero_point data: (0)

input[5]: 
name: points1
valid shape: (10,128,128,2)
aligned byte size: 655360
tensor type: HB_DNN_TENSOR_TYPE_S16
quanti type: SCALE
stride: (65536,512,4,2)
scale data: (0.015625)
zero_point data: (0)

output[0]: 
name: _output_0
valid shape: (1,200,200,16)
aligned byte size: 2560000
tensor type: HB_DNN_TENSOR_TYPE_S32
quanti type: NONE
stride: (2560000,12800,64,4)