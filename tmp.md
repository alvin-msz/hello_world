/root/perception/tros_ai_wrapper/ai_core/code/src/method/qat_bevfusion_multisensor_preprocess_method.cc:695:50: error: ‘struct QATBevFusionMultiSensorPreProcessMethod::OfflineRawConfig’ has no member named ‘lidar2img_npy_name_’; did you mean ‘lidar2img_npy_name’?
  695 |   if (!LoadHomography6FromNpy(path, offline_raw_.lidar2img_npy_name_,
      |                                                  ^~~~~~~~~~~~~~~~~~~
      |                                                  lidar2img_npy_name
[ 18%] Building CXX object CMakeFiles/example.dir/src/method/qat_bevfusion_multitask_post_process_method.cc.o
make[2]: *** [CMakeFiles/example.dir/build.make:454: CMakeFiles/example.dir/src/method/qat_bevfusion_multisensor_preprocess_method.cc.o] Error 1
make[2]: *** Waiting for unfinished jobs....
make[1]: *** [CMakeFiles/Makefile2:83: CMakeFiles/example.dir/all] Error 2
make: *** [Makefile:136: all] Error 2