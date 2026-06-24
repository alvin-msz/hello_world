/usr/bin/ld: CMakeFiles/example.dir/src/method/qat_bevfusion_multisensor_preprocess_method.cc.o: in function `QATBevFusionMultiSensorPreProcessMethod::DoProcessOfflineRaw(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, int,ImageTensor*)':
qat_bevfusion_multisensor_preprocess_method.cc:(.text+0x7ec0): undefined reference to `QATCenterPointPreProcessMethod::DoProcessFromPoints(float const*, int, int, ImageTensor*)'
collect2: error: ld returned 1 exit status
make[2]: *** [CMakeFiles/example.dir/build.make:823: example] Error 1
make[1]: *** [CMakeFiles/Makefile2:83: CMakeFiles/example.dir/all] Error 2
make: *** [Makefile:136: all] Error 2