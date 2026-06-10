// Copyright (c) 2023 Horizon Robotics.All Rights Reserved.
//
// The material in this file is confidential and contains trade secrets
// of Horizon Robotics Inc. This is proprietary information owned by
// Horizon Robotics Inc. No part of this work may be disclosed,
// reproduced, copied, transmitted, or used in any way for any purpose,
// without the express written permission of Horizon Robotics Inc.

#include "method/qat_bevfusion_occ_post_process_method.h"

#include <cmath>
#include <iomanip>
#include <limits>

#include "base/common_def.h"
#include "method/method_data.h"
#include "method/method_factory.h"
#include "rapidjson/document.h"
#include "utils/stop_watch.h"
#include "utils/tensor_utils.h"
#include "utils/utils.h"

DEFINE_AND_REGISTER_METHOD(QATBevFusionOccPostProcessMethod);

namespace {

float QuantScaleAt(const hbDNNTensor *tensor, int32_t idx) {
  const float *sd = tensor->properties.scale.scaleData;
  if (sd == nullptr) {
    return 1.0f;
  }
  const int32_t slen = tensor->properties.scale.scaleLen;
  if (slen <= 0) {
    return 1.0f;
  }
  if (idx >= 0 && idx < slen) {
    return sd[idx];
  }
  // 当 scaleLen == 1 时使用 sd[0]（标量 scale），否则说明索引越界，返回 1.0
  return (slen == 1) ? sd[0] : 1.0f;
}

}  // namespace

int QATBevFusionOccPostProcessMethod::InitFromJsonString(
    const std::string &config) {
  VLOG(EXAMPLE_DEBUG) << "QATBevFusionOccPostProcessMethod Json string:"
                      << config.data();

  rapidjson::Document document;
  document.Parse(config.data());
  if (document.HasParseError()) {
    VLOG(EXAMPLE_SYSTEM) << "Parsing config file failed";
    return -1;
  }

  if (document.HasMember("ori_shape")) {
    auto ori_value = document["ori_shape"].GetArray();
    ori_shape_.resize(ori_value.Size());
    for (rapidjson::SizeType i = 0; i < ori_value.Size(); ++i) {
      ori_shape_[i] = ori_value[i].GetInt();
    }
  }

  if (document.HasMember("resize_shape")) {
    auto resize_value = document["resize_shape"].GetArray();
    resize_shape_.resize(resize_value.Size());
    for (rapidjson::SizeType i = 0; i < resize_value.Size(); ++i) {
      resize_shape_[i] = resize_value[i].GetInt();
    }
  }

  return 0;
}

PerceptionPtr QATBevFusionOccPostProcessMethod::DoProcess(
    ImageTensor *image_tensor, TensorVectorPtr &output_tensor) {
  auto perception = std::shared_ptr<Perception>(new Perception);
  PostProcess(output_tensor->tensors, image_tensor, perception.get());
  return perception;
}

int QATBevFusionOccPostProcessMethod::PostProcess(
    std::vector<hbDNNTensor> &tensors, ImageTensor *image_tensor,
    Perception *perception) {
  if (tensors.empty()) {
    VLOG(EXAMPLE_SYSTEM) << "BEVFusion OCC postprocess: empty output tensors";
    return -1;
  }

  const uint64_t post_t0 = Stopwatch::CurrentTs();
  hbUCPMemFlush(&(tensors[0].sysMem), HB_SYS_MEM_CACHE_INVALIDATE);

  perception->type = Perception::SEG3D;
  image_tensor->ori_image_height = ori_shape_[0];
  image_tensor->ori_image_width = ori_shape_[1];
  image_tensor->resize_height = resize_shape_[0];
  image_tensor->resize_width = resize_shape_[1];

  hbDNNTensor &occ = tensors[0];
  const char *raw_base = reinterpret_cast<const char *>(occ.sysMem.virAddr);

  // Determine element size based on tensor type
  const bool is_s16 = (occ.properties.tensorType == HB_DNN_TENSOR_TYPE_S16);
  const int32_t elem_size = is_s16 ? 2 : 1;

  auto occ_val_at = [&](int32_t pos) -> int32_t {
    if (is_s16) {
      return static_cast<int32_t>(
          *reinterpret_cast<const int16_t *>(raw_base + pos * 2));
    }
    if (occ.properties.tensorType == HB_DNN_TENSOR_TYPE_U8) {
      return static_cast<int32_t>(
          *reinterpret_cast<const uint8_t *>(raw_base + pos));
    }
    return static_cast<int32_t>(
        *reinterpret_cast<const int8_t *>(raw_base + pos));
  };

  int32_t seg_h = occ.properties.validShape.dimensionSize[1];
  int32_t seg_w = occ.properties.validShape.dimensionSize[2];
  int32_t seg_z = occ.properties.validShape.dimensionSize[3];
  int32_t seg_channel = occ.properties.validShape.dimensionSize[4];

  int32_t seg_w_aligned =
      occ.properties.stride[1] / occ.properties.stride[2];
  int32_t seg_z_aligned =
      occ.properties.stride[2] / occ.properties.stride[3];
  int32_t seg_channel_aligned =
      occ.properties.stride[3] / occ.properties.stride[4];

  VLOG(EXAMPLE_DEBUG) << "OCC output shape [" << seg_h << "," << seg_w << ","
                      << seg_z << "," << seg_channel << "]";

  int32_t seg_k_hwz = seg_h * seg_w * seg_z;
  perception->seg3d.num_classes = seg_channel;
  perception->seg3d.seg.resize(static_cast<size_t>(seg_k_hwz));
  perception->seg3d.h = seg_h;
  perception->seg3d.w = seg_w;
  perception->seg3d.z = seg_z;

  for (int32_t h = 0; h < seg_h; ++h) {
    int32_t h_idx = h * seg_w_aligned * seg_z_aligned * seg_channel_aligned;
    int32_t h_k = h * seg_w * seg_z;
    for (int32_t w = 0; w < seg_w; ++w) {
      int32_t hw_idx = h_idx + w * seg_z_aligned * seg_channel_aligned;
      int32_t hw_k = h_k + w * seg_z;
      for (int32_t z = 0; z < seg_z; ++z) {
        int32_t idx = hw_idx + z * seg_channel_aligned;
        int32_t k = hw_k + z;
        float max_val = std::numeric_limits<float>::lowest();
        int32_t top_index = -1;
        for (int c = 0; c < seg_channel; ++c) {
          float sc = QuantScaleAt(&occ, c);
          float data = quanti_scale(occ_val_at(idx + c), sc);
          if (data > max_val) {
            max_val = data;
            top_index = c;
          }
        }
        perception->seg3d.seg[static_cast<size_t>(k)] =
            static_cast<uint32_t>(std::max(0, top_index));
      }
    }
  }

  const uint64_t post_us = Stopwatch::CurrentTs() - post_t0;
  VLOG(EXAMPLE_REPORT) << std::fixed << std::setprecision(3)
                       << "BevFusion OCC latency frame_id="
                       << image_tensor->frame_id << " pre_ms="
                       << image_tensor->pre_duration / 1000.0 << " infer_ms="
                       << image_tensor->infer_duration / 1000.0 << " post_ms="
                       << post_us / 1000.0;
  return 0;
}
