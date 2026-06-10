/root/perception/tros_ai_wrapper/ai_core/code/src/method/qat_bevfusion_occ_4d_post_process_method.cc: In member function ‘virtual int QATBevFusionOccPostProcessMethod::InitFromJsonString(const string&)’:
/root/perception/tros_ai_wrapper/ai_core/code/src/method/qat_bevfusion_occ_4d_post_process_method.cc:62:5: error: ‘num_classes_’ was not declared in this scope
   62 |     num_classes_ = document["num_classes"].GetInt();
      |     ^~~~~~~~~~~~
/root/perception/tros_ai_wrapper/ai_core/code/src/method/qat_bevfusion_occ_4d_post_process_method.cc: In member function ‘int QATBevFusionOccPostProcessMethod::PostProcess(std::vector<hbDNNTensor>&, ImageTensor*, Perception*)’:
/root/perception/tros_ai_wrapper/ai_core/code/src/method/qat_bevfusion_occ_4d_post_process_method.cc:104:9: error: ‘quant_data_’ was not declared in this scope
  104 |     if (quant_data_.virAddr == nullptr) {
      |         ^~~~~~~~~~~
/root/perception/tros_ai_wrapper/ai_core/code/src/method/qat_bevfusion_occ_4d_post_process_method.cc:108:55: error: ‘quant_data_’ was not declared in this scope; did you mean ‘quant_data_ptr’?
  108 |     float *quant_data_ptr = reinterpret_cast<float *>(quant_data_.virAddr);
      |                                                       ^~~~~~~~~~~
      |                                                       quant_data_ptr
/root/perception/tros_ai_wrapper/ai_core/code/src/method/qat_bevfusion_occ_4d_post_process_method.cc:156:57: error: ‘num_classes_’ was not declared in this scope
  156 |   perception->seg3d.num_classes = static_cast<uint32_t>(num_classes_);
      |                                                         ^~~~~~~~~~~~
make[2]: *** [CMakeFiles/example.dir/build.make:468: CMakeFiles/example.dir/src/method/qat_bevfusion_occ_4d_post_process_method.cc.o] Error 1
make[2]: *** Waiting for unfinished jobs....
make[1]: *** [CMakeFiles/Makefile2:83: CMakeFiles/example.dir/all] Error 2
make: *** [Makefile:136: all] Error 2