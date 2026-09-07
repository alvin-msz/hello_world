/root/perception/tros_ai_wrapper/ai_core/code/src/method/qat_bevfusion_multitask_post_process_method.cc: In function ‘void {anonymous}::OccCopyNdhwClassIds(hbDNNTensor*, Parsing3d<unsigned int>*, Parsing<unsigned char>*, int32_t, int32_t, int32_t)’:
/root/perception/tros_ai_wrapper/ai_core/code/src/method/qat_bevfusion_multitask_post_process_method.cc:294:46: error: cannot convert ‘int64_t*’ {aka ‘long int*’} to ‘const int32_t*’ {aka ‘const int*’} in initialization
  294 |   const int32_t *stride = tensor->properties.stride;
      |                           ~~~~~~~~~~~~~~~~~~~^~~~~~
      |                                              |
      |                                              int64_t* {aka long int*}
[ 67%] Building CXX object CMakeFiles/example.dir/src/method/qat_bevfusion_occ_4d_post_process_method.cc.o
make[2]: *** [CMakeFiles/example.dir/build.make:454: CMakeFiles/example.dir/src/method/qat_bevfusion_multitask_post_process_method.cc.o] Error 1
make[2]: *** Waiting for unfinished jobs....
make[1]: *** [CMakeFiles/Makefile2:83: CMakeFiles/example.dir/all] Error 2