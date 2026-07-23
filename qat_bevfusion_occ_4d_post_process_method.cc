// Copyright (c) 2023 Horizon Robotics.All Rights Reserved.
//
// The material in this file is confidential and contains trade secrets
// of Horizon Robotics Inc. This is proprietary information owned by
// Horizon Robotics Inc. No part of this work may be disclosed,
// reproduced, copied, transmitted, or used in any way for any purpose,
// without the express written permission of Horizon Robotics Inc.

#include "method/qat_bevfusion_occ_4d_post_process_method.h"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <sys/stat.h>

#include "base/common_def.h"
#include "method/method_data.h"
#include "method/method_factory.h"
#include "plugin/workflow_plugin.h"
#include "rapidjson/document.h"
#include "utils/stop_watch.h"
#include "utils/tensor_utils.h"
#include "utils/utils.h"

DEFINE_AND_REGISTER_METHOD(QATBevFusionOcc4dPostProcessMethod);

namespace {

constexpr int32_t kOccOutputIndex = 0;
constexpr int32_t kPrevBevOutputIndex = 1;

}  // namespace

int QATBevFusionOcc4dPostProcessMethod::InitFromJsonString(
    const std::string &config) {
  VLOG(EXAMPLE_DEBUG) << "QATBevFusionOcc4dPostProcessMethod Json string:"
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

  if (document.HasMember("num_classes")) {
    num_classes_ = document["num_classes"].GetInt();
  }

  if (document.HasMember("eval_output_dir")) {
    eval_output_dir_ = document["eval_output_dir"].GetString();
    if (!eval_output_dir_.empty()) {
      struct stat st {};
      if (stat(eval_output_dir_.c_str(), &st) != 0) {
        if (mkdir(eval_output_dir_.c_str(), 0755) != 0) {
          VLOG(EXAMPLE_SYSTEM) << "Create eval_output_dir failed: "
                               << eval_output_dir_;
          return -1;
        }
      } else if (!S_ISDIR(st.st_mode)) {
        VLOG(EXAMPLE_SYSTEM) << "eval_output_dir is not a directory: "
                             << eval_output_dir_;
        return -1;
      }
    }
  }

  if (document.HasMember("eval_bin_prefix")) {
    eval_bin_prefix_ = document["eval_bin_prefix"].GetString();
  }

  return 0;
}

PerceptionPtr QATBevFusionOcc4dPostProcessMethod::DoProcess(
    ImageTensor *image_tensor, TensorVectorPtr &output_tensor) {
  auto perception = std::shared_ptr<Perception>(new Perception);
  PostProcess(output_tensor->tensors, image_tensor, perception.get());
  return perception;
}

int QATBevFusionOcc4dPostProcessMethod::PostProcess(
    std::vector<hbDNNTensor> &tensors, ImageTensor *image_tensor,
    Perception *perception) {
  if (tensors.empty()) {
    VLOG(EXAMPLE_SYSTEM) << "BEVFusion OCC postprocess: empty output tensors";
    return -1;
  }

  const uint64_t post_t0 = Stopwatch::CurrentTs();

  bool is_temporal = WorkflowPlugin::GetInstance()->IsTemporalModel();
  VLOG(EXAMPLE_DEBUG) << "Is temporal model: " << is_temporal
                      << "; output tensor count:" << tensors.size();

  if (is_temporal && tensors.size() > kPrevBevOutputIndex) {
    hbDNNTensor &prev_bev = tensors[kPrevBevOutputIndex];
    hbUCPMemFlush(&(prev_bev.sysMem), HB_SYS_MEM_CACHE_INVALIDATE);

    int32_t associated_prev_num =
        WorkflowPlugin::GetInstance()->GetAssociatedPrevNum();
    const int16_t *bev_data =
        reinterpret_cast<const int16_t *>(prev_bev.sysMem.virAddr);
    float scale = 1.0f;
    if (prev_bev.properties.scale.scaleData != nullptr) {
      scale = prev_bev.properties.scale.scaleData[0];
    }

    const int32_t output_elems =
        prev_bev.properties.alignedByteSize / sizeof(int16_t);
    if (quant_data_.virAddr == nullptr) {
      hbUCPMallocCached(&quant_data_,
                        static_cast<uint64_t>(output_elems) * sizeof(float), 0);
    }
    float *quant_data_ptr = reinterpret_cast<float *>(quant_data_.virAddr);
    for (int32_t i = 0; i < output_elems; ++i) {
      quant_data_ptr[i] = static_cast<float>(bev_data[i]) * scale;
    }
    WorkflowPlugin::GetInstance()->UpdateTemporalQuantTensor<float *>(
        &quant_data_, associated_prev_num - 1, 0);
  }

  hbUCPMemFlush(&(tensors[kOccOutputIndex].sysMem),
                HB_SYS_MEM_CACHE_INVALIDATE);

  perception->type = Perception::SEG3D;
  image_tensor->ori_image_height = ori_shape_[0];
  image_tensor->ori_image_width = ori_shape_[1];
  image_tensor->resize_height = resize_shape_[0];
  image_tensor->resize_width = resize_shape_[1];

  hbDNNTensor &occ = tensors[kOccOutputIndex];
  auto &vs = occ.properties.validShape;
  const int32_t num_dims = vs.numDimensions;

  int32_t seg_h = 0;
  int32_t seg_w = 0;
  int32_t seg_z = 0;
  int32_t seg_w_aligned = 0;
  int32_t seg_z_aligned = 0;

  if (num_dims == 4) {
    seg_h = vs.dimensionSize[1];
    seg_w = vs.dimensionSize[2];
    seg_z = vs.dimensionSize[3];
    seg_w_aligned = occ.properties.stride[1] / occ.properties.stride[2];
    seg_z_aligned = occ.properties.stride[2] / occ.properties.stride[3];
  } else if (num_dims == 5) {
    seg_h = vs.dimensionSize[1];
    seg_w = vs.dimensionSize[2];
    seg_z = vs.dimensionSize[3];
    seg_w_aligned = occ.properties.stride[1] / occ.properties.stride[2];
    seg_z_aligned = occ.properties.stride[2] / occ.properties.stride[3];
  } else {
    VLOG(EXAMPLE_SYSTEM) << "Unsupported OCC output rank: " << num_dims;
    return -1;
  }

  VLOG(EXAMPLE_DEBUG) << "OCC output shape [" << seg_h << "," << seg_w << ","
                      << seg_z << "] type=" << occ.properties.tensorType;

  const int32_t seg_k_hwz = seg_h * seg_w * seg_z;
  perception->seg3d.num_classes = static_cast<uint32_t>(num_classes_);
  perception->seg3d.seg.resize(static_cast<size_t>(seg_k_hwz));
  perception->seg3d.h = seg_h;
  perception->seg3d.w = seg_w;
  perception->seg3d.z = seg_z;

  if (occ.properties.tensorType == HB_DNN_TENSOR_TYPE_S32 && num_dims == 4) {
    const int32_t *occ_data =
        reinterpret_cast<const int32_t *>(occ.sysMem.virAddr);
    for (int32_t h = 0; h < seg_h; ++h) {
      const int32_t h_idx = h * seg_w_aligned * seg_z_aligned;
      const int32_t h_k = h * seg_w * seg_z;
      for (int32_t w = 0; w < seg_w; ++w) {
        const int32_t hw_idx = h_idx + w * seg_z_aligned;
        const int32_t hw_k = h_k + w * seg_z;
        for (int32_t z = 0; z < seg_z; ++z) {
          const int32_t idx = hw_idx + z;
          const int32_t k = hw_k + z;
          const int32_t label = occ_data[idx];
          perception->seg3d.seg[static_cast<size_t>(k)] =
              static_cast<uint32_t>(std::max(0, label));
        }
      }
    }
  } else {
    const char *raw_base = reinterpret_cast<const char *>(occ.sysMem.virAddr);
    const bool is_s16 = (occ.properties.tensorType == HB_DNN_TENSOR_TYPE_S16);
    const int32_t seg_channel =
        (num_dims == 5) ? vs.dimensionSize[4] : num_classes_;
    const int32_t seg_channel_aligned =
        (num_dims == 5) ? occ.properties.stride[3] / occ.properties.stride[4]
                        : 1;

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

    for (int32_t h = 0; h < seg_h; ++h) {
      const int32_t h_idx =
          h * seg_w_aligned * seg_z_aligned * seg_channel_aligned;
      const int32_t h_k = h * seg_w * seg_z;
      for (int32_t w = 0; w < seg_w; ++w) {
        const int32_t hw_idx = h_idx + w * seg_z_aligned * seg_channel_aligned;
        const int32_t hw_k = h_k + w * seg_z;
        for (int32_t z = 0; z < seg_z; ++z) {
          const int32_t idx = hw_idx + z * seg_channel_aligned;
          const int32_t k = hw_k + z;
          float max_val = std::numeric_limits<float>::lowest();
          int32_t top_index = -1;
          for (int c = 0; c < seg_channel; ++c) {
            const float *sd = occ.properties.scale.scaleData;
            const float sc =
                (sd != nullptr) ? ((occ.properties.scale.scaleLen == 1)
                                       ? sd[0]
                                       : sd[c])
                                : 1.0f;
            const float data = quanti_scale(occ_val_at(idx + c), sc);
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
  }

  if (!eval_output_dir_.empty()) {
    SaveOccPredBin(image_tensor, perception);
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

int QATBevFusionOcc4dPostProcessMethod::SaveOccPredBin(
    const ImageTensor *image_tensor, const Perception *perception) {
  if (perception->seg3d.seg.empty() || perception->seg3d.h <= 0 ||
      perception->seg3d.w <= 0 || perception->seg3d.z <= 0) {
    VLOG(EXAMPLE_SYSTEM) << "Skip OCC bin export: empty seg3d result, frame_id="
                         << image_tensor->frame_id;
    return -1;
  }

  std::ostringstream oss;
  oss << eval_output_dir_ << "/" << eval_bin_prefix_ << std::setw(6)
      << std::setfill('0') << image_tensor->frame_id << ".bin";
  const std::string bin_path = oss.str();

  const size_t elem_count = perception->seg3d.seg.size();
  std::vector<int16_t> pred(elem_count);
  for (size_t i = 0; i < elem_count; ++i) {
    const int32_t label = static_cast<int32_t>(perception->seg3d.seg[i]);
    pred[i] = static_cast<int16_t>(std::max(0, std::min(label, 32767)));
  }

  std::ofstream ofs(bin_path, std::ios::out | std::ios::binary | std::ios::trunc);
  if (!ofs) {
    VLOG(EXAMPLE_SYSTEM) << "Open OCC pred bin failed: " << bin_path;
    return -1;
  }
  ofs.write(reinterpret_cast<const char *>(pred.data()),
            static_cast<std::streamsize>(pred.size() * sizeof(int16_t)));
  if (!ofs) {
    VLOG(EXAMPLE_SYSTEM) << "Write OCC pred bin failed: " << bin_path;
    return -1;
  }

  VLOG(EXAMPLE_DEBUG) << "Saved OCC pred bin: " << bin_path
                      << " shape=[" << perception->seg3d.h << ","
                      << perception->seg3d.w << "," << perception->seg3d.z
                      << "] elems=" << elem_count;
  return 0;
}
