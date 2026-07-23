// Copyright (c) 2023 Horizon Robotics.All Rights Reserved.
//
// The material in this file is confidential and contains trade secrets
// of Horizon Robotics Inc. This is proprietary information owned by
// Horizon Robotics Inc. No part of this work may be disclosed,
// reproduced, copied, transmitted, or used in any way for any purpose,
// without the express written permission of Horizon Robotics Inc.

#ifndef DNN_AI_BENCHMARK_CODE_INCLUDE_METHOD_QAT_BEVFUSION_OCC_POST_PROCESS_METHOD_H_
#define DNN_AI_BENCHMARK_CODE_INCLUDE_METHOD_QAT_BEVFUSION_OCC_POST_PROCESS_METHOD_H_

#include <string>
#include <vector>

#include "glog/logging.h"
#include "method/method_data.h"
#include "method/post_process_method.h"

/**
 * Post process for BEVFusion OCC temporal model:
 *   output 0 occ      [1, 200, 200, 16] INT32  (per-voxel class id)
 *   output 1 prev_bev [1, 64, 200, 200] INT16  (temporal state)
 */
class QATBevFusionOcc4dPostProcessMethod : public PostProcessMethod {
 public:
  int InitFromJsonString(const std::string &config) override;

  PerceptionPtr DoProcess(ImageTensor *image_tensor,
                          TensorVectorPtr &output_tensor) override;

  ~QATBevFusionOcc4dPostProcessMethod() override {
    if (quant_data_.virAddr != nullptr) {
      hbUCPFree(&quant_data_);
    }
  }

 private:
  int PostProcess(std::vector<hbDNNTensor> &tensors, ImageTensor *image_tensor,
                  Perception *perception);
  int SaveOccPredBin(const ImageTensor *image_tensor,
                     const Perception *perception);

  std::vector<int> ori_shape_{900, 1600};
  std::vector<int> resize_shape_{256, 704};
  int32_t num_classes_{18};
  std::string eval_output_dir_;
  std::string eval_bin_prefix_{"occ_rank0_"};
  hbUCPSysMem quant_data_{0, nullptr, 0};
};

#endif  // DNN_AI_BENCHMARK_CODE_INCLUDE_METHOD_QAT_BEVFUSION_OCC_POST_PROCESS_METHOD_H_
