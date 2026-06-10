// Copyright (c) 2023 Horizon Robotics.All Rights Reserved.
//
// The material in this file is confidential and contains trade secrets
// of Horizon Robotics Inc. This is proprietary information owned by
// Horizon Robotics Inc. No part of this work may be disclosed,
// reproduced, copied, transmitted, or used in any way for any purpose,
// without the express written permission of Horizon Robotics Inc.

#ifndef DNN_AI_BENCHMARK_CODE_INCLUDE_METHOD_QAT_BEVFUSION_OCC_PREPROCESS_METHOD_H_
#define DNN_AI_BENCHMARK_CODE_INCLUDE_METHOD_QAT_BEVFUSION_OCC_PREPROCESS_METHOD_H_

#include <cstdint>
#include <string>
#include <vector>

#include "glog/logging.h"
#include "method/method_data.h"
#include "method/preprocess_method.h"

/**
 * Preprocess for BEVFusion OCC (camera + lidar), aligned with
 * flashocc-r50-M0_bevfusionocc_horizon_2 / HBM inputs:
 *   0 img          [1, 6, 3, 256, 704]   INT8
 *   1 post_rot     [1, 6, 3, 3]          INT8
 *   2 bda          [1, 3, 3]             FLOAT32
 *   3 sensor2ego   [1, 6, 4, 4]          INT8
 *   4 points       [30000, 4]            FLOAT32
 *
 * Each line in the raw_data list file is a directory containing per-input bins
 * (see auxiliary_bin_names and lidar_bin_name).
 */
class QATBevFusionOccPreProcessMethod : public PreProcessMethod {
 public:
  int32_t InitFromJsonString(const std::string &config) override;

  int32_t DoProcess(std::string path, int32_t input_count,
                    ImageTensor *image_tensor) override;

  ~QATBevFusionOccPreProcessMethod() override = default;

 private:
  static bool IsDirectory(const std::string &path);
  static std::string JoinPath(const std::string &dir, const std::string &name);
  static bool TryLoadExactAlignedBin(hbDNNTensor *tensor,
                                     const std::string &bin_path);

  int32_t LoadFloatInputFromBin(hbDNNTensor *tensor, const std::string &bin_path,
                                uint64_t *preprocess_us);
  int32_t LoadAuxiliaryInput(hbDNNTensor *tensor, const std::string &bin_path,
                             uint64_t *preprocess_us);
  int32_t LoadPointsInput(hbDNNTensor *tensor, const std::string &bin_path,
                          uint64_t *preprocess_us);

  std::vector<std::string> auxiliary_bin_names_{
      "img.bin", "post_rot.bin", "bda.bin", "sensor2ego.bin"};
  std::string lidar_bin_name_{"lidar_points.bin"};
  std::vector<float> point_cloud_range_{-40.f, -40.f, -1.f, 40.f, 40.f, 5.4f};
  int32_t max_num_points_{30000};
  int32_t point_dim_{4};
};

#endif  // DNN_AI_BENCHMARK_CODE_INCLUDE_METHOD_QAT_BEVFUSION_OCC_PREPROCESS_METHOD_H_
