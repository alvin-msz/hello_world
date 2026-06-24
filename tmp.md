/root/perception/tros_ai_wrapper/ai_core/code/src/method/qat_bevfusion_multisensor_preprocess_method.cc: In member function ‘bool QATBevFusionMultiSensorPreProcessMethod::IsOfflineRawSampleDir(const string&) const’:
/root/perception/tros_ai_wrapper/ai_core/code/src/method/qat_bevfusion_multisensor_preprocess_method.cc:283:52: error: ‘const struct QATBevFusionMultiSensorPreProcessMethod::OfflineRawConfig’ has no member named ‘lidar_npy_name_’; did you mean ‘lidar_npy_name’?
  283 |   return IsRegularFile(JoinPath(path, offline_raw_.lidar_npy_name_));
      |                                                    ^~~~~~~~~~~~~~~
      |                                                    lidar_npy_name
/root/perception/tros_ai_wrapper/ai_core/code/src/method/qat_bevfusion_multisensor_preprocess_method.cc: In member function ‘bool QATBevFusionMultiSensorPreProcessMethod::LoadLidarFromNpy(const string&, std::vector<float>*) const’:
/root/perception/tros_ai_wrapper/ai_core/code/src/method/qat_bevfusion_multisensor_preprocess_method.cc:289:41: error: ‘const struct QATBevFusionMultiSensorPreProcessMethod::OfflineRawConfig’ has no member named ‘lidar_npy_name_’; did you mean ‘lidar_npy_name’?
  289 |       JoinPath(sample_dir, offline_raw_.lidar_npy_name_);
      |                                         ^~~~~~~~~~~~~~~
      |                                         lidar_npy_name
/root/perception/tros_ai_wrapper/ai_core/code/src/method/qat_bevfusion_multisensor_preprocess_method.cc: In member function ‘bool QATBevFusionMultiSensorPreProcessMethod::LoadCameraJpgBuffersFromDir(const string&, std::vector<CameraJpgBuffer>*) const’:
/root/perception/tros_ai_wrapper/ai_core/code/src/method/qat_bevfusion_multisensor_preprocess_method.cc:401:27: error: no matching function for call to ‘std::vector<CameraJpgBuffer>::push_back(<brace-enclosed initializer list>)’
  401 |     jpg_buffers->push_back({cam_idx, bytes});
      |     ~~~~~~~~~~~~~~~~~~~~~~^~~~~~~~~~~~~~~~~~
In file included from /usr/include/c++/11/vector:67,
                 from /root/perception/tros_ai_wrapper/ai_core/code/include/method/qat_bevfusion_multisensor_preprocess_method.h:14,
                 from /root/perception/tros_ai_wrapper/ai_core/code/src/method/qat_bevfusion_multisensor_preprocess_method.cc:9:
/usr/include/c++/11/bits/stl_vector.h:1187:7: note: candidate: ‘void std::vector<_Tp, _Alloc>::push_back(const value_type&) [with _Tp = CameraJpgBuffer; _Alloc = std::allocator<CameraJpgBuffer>; std::vector<_Tp, _Alloc>::value_type = CameraJpgBuffer]’
 1187 |       push_back(const value_type& __x)
      |       ^~~~~~~~~
/usr/include/c++/11/bits/stl_vector.h:1187:35: note:   no known conversion for argument 1 from ‘<brace-enclosed initializer list>’ to ‘const value_type&’ {aka ‘const CameraJpgBuffer&’}
 1187 |       push_back(const value_type& __x)
      |                 ~~~~~~~~~~~~~~~~~~^~~
/usr/include/c++/11/bits/stl_vector.h:1203:7: note: candidate: ‘void std::vector<_Tp, _Alloc>::push_back(std::vector<_Tp, _Alloc>::value_type&&) [with _Tp = CameraJpgBuffer; _Alloc = std::allocator<CameraJpgBuffer>; std::vector<_Tp, _Alloc>::value_type = CameraJpgBuffer]’
 1203 |       push_back(value_type&& __x)
      |       ^~~~~~~~~
/usr/include/c++/11/bits/stl_vector.h:1203:30: note:   no known conversion for argument 1 from ‘<brace-enclosed initializer list>’ to ‘std::vector<CameraJpgBuffer>::value_type&&’ {aka ‘CameraJpgBuffer&&’}
 1203 |       push_back(value_type&& __x)
      |                 ~~~~~~~~~~~~~^~~
make[2]: *** [CMakeFiles/example.dir/build.make:454: CMakeFiles/example.dir/src/method/qat_bevfusion_multisensor_preprocess_method.cc.o] Error 1
make[2]: *** Waiting for unfinished jobs....
make[1]: *** [CMakeFiles/Makefile2:83: CMakeFiles/example.dir/all] Error 2
make: *** [Makefile:136: all] Error 2