/root/perception/tros_ai_wrapper/ai_core/code/src/method/bevfusion_vt_auxiliary.cc: In function ‘bool {anonymous}::PointSampling(const std::vector<float>&, const float*, const Homography6*, int, int, const BevFusionVtConfig&, BevFusionAuxTensors*)’:
/root/perception/tros_ai_wrapper/ai_core/code/src/method/bevfusion_vt_auxiliary.cc:108:52: error: cannot convert ‘std::array<std::array<float, 4>, 4>::const_pointer’ {aka ‘const std::array<float, 4>*’} to ‘const float (*)[4]’ in initialization
  108 |     const float (*ego2img)[4] = lidar2img[cam].data();
      |                                 ~~~~~~~~~~~~~~~~~~~^~
      |                                                    |
      |                                                    std::array<std::array<float, 4>, 4>::const_pointer {aka const std::array<float, 4>*}
/root/perception/tros_ai_wrapper/ai_core/code/src/method/bevfusion_vt_auxiliary.cc: In function ‘void BevFusionHomographyResizeAndCrop(Homography6*, int, int, int, int)’:
/root/perception/tros_ai_wrapper/ai_core/code/src/method/bevfusion_vt_auxiliary.cc:312:38: error: cannot convert ‘std::array<std::array<float, 4>, 4>::pointer’ {aka ‘std::array<float, 4>*’} to ‘const float (*)[4]’
  312 |     Mat4Mul(view, lidar2img[cam].data(), tmp);
      |                   ~~~~~~~~~~~~~~~~~~~^~
      |                                      |
      |                                      std::array<std::array<float, 4>, 4>::pointer {aka std::array<float, 4>*}
/root/perception/tros_ai_wrapper/ai_core/code/src/method/bevfusion_vt_auxiliary.cc:25:47: note:   initializing argument 2 of ‘void {anonymous}::Mat4Mul(const float (*)[4], const float (*)[4], float (*)[4])’
   25 | void Mat4Mul(const float a[4][4], const float b[4][4], float out[4][4]) {
      |                                   ~~~~~~~~~~~~^~~~~~~
make[2]: *** [CMakeFiles/example.dir/build.make:356: CMakeFiles/example.dir/src/method/bevfusion_vt_auxiliary.cc.o] Error 1
make[2]: *** Waiting for unfinished jobs....
make[1]: *** [CMakeFiles/Makefile2:83: CMakeFiles/example.dir/all] Error 2
make: *** [Makefile:136: all] Error 2