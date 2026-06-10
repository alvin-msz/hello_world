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
 * Post process for BEVFusion OCC: single output [1, H, W, Dz, C] (e.g.
 * [1, 200, 200, 16, 18] INT8), per-voxel argmax over classes (BEVOCCHead2D).
 */
class QATBevFusionOccPostProcessMethod : public PostProcessMethod {
 public:
  int InitFromJsonString(const std::string &config) override;

  PerceptionPtr DoProcess(ImageTensor *image_tensor,
                          TensorVectorPtr &output_tensor) override;

  ~QATBevFusionOccPostProcessMethod() override = default;

 private:
  int PostProcess(std::vector<hbDNNTensor> &tensors, ImageTensor *image_tensor,
                  Perception *perception);

  std::vector<int> ori_shape_{900, 1600};
  std::vector<int> resize_shape_{256, 704};
};

#endif  // DNN_AI_BENCHMARK_CODE_INCLUDE_METHOD_QAT_BEVFUSION_OCC_POST_PROCESS_METHOD_H_
