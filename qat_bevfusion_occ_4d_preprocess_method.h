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
 * Preprocess for BEVFusion OCC temporal model (camera + lidar), aligned with
 * flashocc-r50-M0_bevfusionocc_horizon_2 / HBM inputs:
 *   0 imgs         [1, 6, 3, 256, 704]   INT16
 *   1 uv           [1, 6, 320000, 2]      INT16
 *   2 zcam         [1, 6, 320000]         INT16
 *   3 points       [30000, 4]             FLOAT32
 *   4 prev_bev     [1, 64, 200, 200]       INT8
 *
 * Each line in the raw_data list file is a directory containing five bins:
 * imgs.bin, uv.bin, zcam.bin, points.bin, prev_bev.bin.
 * prev_bev.bin may be INT8 (aligned with input_4) or INT16 (aligned with
 * output_1); INT16 sources are dequantized with prev_bev_src_scale and
 * re-quantized to INT8. At runtime, WorkflowPlugin may overwrite input_4
 * with the converted output_1 from the previous inference frame.
 */
class QATBevFusionOcc4dPreProcessMethod : public PreProcessMethod {
 public:
  int32_t InitFromJsonString(const std::string &config) override;

  int32_t DoProcess(std::string path, int32_t input_count,
                    ImageTensor *image_tensor) override;

  ~QATBevFusionOcc4dPreProcessMethod() override = default;

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
  int32_t LoadPrevBevInput(hbDNNTensor *tensor, const std::string &bin_path,
                           uint64_t *preprocess_us);

  std::vector<std::string> auxiliary_bin_names_{
      "imgs.bin", "uv.bin", "zcam.bin"};
  std::string lidar_bin_name_{"points.bin"};
  std::string prev_bev_bin_name_{"prev_bev.bin"};
  float prev_bev_src_scale_{0.f};
  std::vector<float> point_cloud_range_{-40.f, -40.f, -1.f, 40.f, 40.f, 5.4f};
  int32_t max_num_points_{30000};
  int32_t point_dim_{4};
};

#endif  // DNN_AI_BENCHMARK_CODE_INCLUDE_METHOD_QAT_BEVFUSION_OCC_PREPROCESS_METHOD_H_
