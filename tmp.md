qat_bevfusion_occ_post_process_method.cc: In member function ‘virtual int QATBevFusionOccPostProcessMethod::InitFromJsonString(const string&)’:
/root/perception/tros_ai_wrapper/ai_core/code/src/method/qat_bevfusion_occ_post_process_method.cc:64:5: error: ‘num_classes_’ was not declared in this scope
   64 |     num_classes_ = document["num_classes"].GetInt();
      |     ^~~~~~~~~~~~
/root/perception/tros_ai_wrapper/ai_core/code/src/method/qat_bevfusion_occ_post_process_method.cc:74:5: error: ‘eval_bin_prefix_’ was not declared in this scope
   74 |     eval_bin_prefix_ = document["eval_bin_prefix"].GetString();
      |     ^~~~~~~~~~~~~~~~
/root/perception/tros_ai_wrapper/ai_core/code/src/method/qat_bevfusion_occ_post_process_method.cc: In member function ‘int QATBevFusionOccPostProcessMethod::PostProcess(std::vector<hbDNNTensor>&, ImageTensor*, Perception*)’:
/root/perception/tros_ai_wrapper/ai_core/code/src/method/qat_bevfusion_occ_post_process_method.cc:133:57: error: ‘num_classes_’ was not declared in this scope
  133 |   perception->seg3d.num_classes = static_cast<uint32_t>(num_classes_);
      |                                                         ^~~~~~~~~~~~
/root/perception/tros_ai_wrapper/ai_core/code/src/method/qat_bevfusion_occ_post_process_method.cc:158:5: error: ‘SaveOccPredBin’ was not declared in this scope
  158 |     SaveOccPredBin(image_tensor, perception);
      |     ^~~~~~~~~~~~~~
/root/perception/tros_ai_wrapper/ai_core/code/src/method/qat_bevfusion_occ_post_process_method.cc: At global scope:
/root/perception/tros_ai_wrapper/ai_core/code/src/method/qat_bevfusion_occ_post_process_method.cc:171:5: error: no declaration matches ‘int QATBevFusionOccPostProcessMethod::SaveOccPredBin(const ImageTensor*, const Perception*)’
  171 | int QATBevFusionOccPostProcessMethod::SaveOccPredBin(
      |     ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
/root/perception/tros_ai_wrapper/ai_core/code/src/method/qat_bevfusion_occ_post_process_method.cc:171:5: note: no functions named ‘int QATBevFusionOccPostProcessMethod::SaveOccPredBin(const ImageTensor*, const Perception*)’
In file included from /root/perception/tros_ai_wrapper/ai_core/code/src/method/qat_bevfusion_occ_post_process_method.cc:9:
/root/perception/tros_ai_wrapper/ai_core/code/include/method/qat_bevfusion_occ_post_process_method.h:24:7: note: ‘class QATBevFusionOccPostProcessMethod’ defined here
   24 | class QATBevFusionOccPostProcessMethod : public PostProcessMethod {
      |       ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
make[2]: *** [CMakeFiles/example.dir/build.make:496: CMakeFiles/example.dir/src/method/qat_bevfusion_occ_post_process_method.cc.o] Error 1
make[2]: *** Waiting for unfinished jobs....
make[1]: *** [CMakeFiles/Makefile2:83: CMakeFiles/example.dir/all] Error 2
make: *** [Makefile:136: all] Error 2