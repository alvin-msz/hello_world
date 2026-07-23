// Copyright (c) 2023 Horizon Robotics.All Rights Reserved.
//
// The material in this file is confidential and contains trade secrets
// of Horizon Robotics Inc. This is proprietary information owned by
// Horizon Robotics Inc. No part of this work may be disclosed,
// reproduced, copied, transmitted, or used in any way for any purpose,
// without the express written permission of Horizon Robotics Inc.

#include "method/qat_bevfusion_occ_4d_preprocess_method.h"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <sys/stat.h>

#include "base/common_def.h"
#include "method/method_factory.h"
#include "rapidjson/document.h"
#include "utils/stop_watch.h"
#include "utils/tensor_utils.h"

DEFINE_AND_REGISTER_METHOD(QATBevFusionOcc4dPreProcessMethod);

namespace {

constexpr int32_t kExpectedCameraAuxInputs = 3;
constexpr int32_t kPointsInputIndex = 3;
constexpr int32_t kPrevBevInputIndex = 4;
constexpr int32_t kExpectedInputCount = 5;

}  // namespace

bool QATBevFusionOcc4dPreProcessMethod::IsDirectory(const std::string &path) {
  struct stat st {};
  if (stat(path.c_str(), &st) != 0) {
    return false;
  }
  return S_ISDIR(st.st_mode);
}

std::string QATBevFusionOcc4dPreProcessMethod::JoinPath(const std::string &dir,
                                                      const std::string &name) {
  if (dir.empty()) {
    return name;
  }
  if (!name.empty() && name.front() == '/') {
    return name;
  }
  if (dir.back() == '/') {
    return dir + name;
  }
  return dir + "/" + name;
}

bool QATBevFusionOcc4dPreProcessMethod::TryLoadExactAlignedBin(
    hbDNNTensor *tensor, const std::string &bin_path) {
  std::ifstream ifs(bin_path.c_str(), std::ios::in | std::ios::binary);
  if (!ifs) {
    return false;
  }
  ifs.seekg(0, std::ios::end);
  auto len = static_cast<size_t>(ifs.tellg());
  ifs.seekg(0, std::ios::beg);
  size_t aligned = static_cast<size_t>(tensor->properties.alignedByteSize);
  if (len != aligned) {
    return false;
  }
  ifs.read(reinterpret_cast<char *>(tensor->sysMem.virAddr),
           static_cast<std::streamsize>(len));
  hbUCPMemFlush(&tensor->sysMem, HB_SYS_MEM_CACHE_CLEAN);
  return !ifs.fail() && static_cast<size_t>(ifs.gcount()) == len;
}

int32_t QATBevFusionOcc4dPreProcessMethod::LoadFloatInputFromBin(
    hbDNNTensor *tensor, const std::string &bin_path, uint64_t *preprocess_us) {
  std::ifstream ifs(bin_path.c_str(), std::ios::in | std::ios::binary);
  if (!ifs) {
    VLOG(EXAMPLE_SYSTEM) << "Open float bin failed: " << bin_path;
    return -1;
  }
  ifs.seekg(0, std::ios::end);
  const size_t file_bytes = static_cast<size_t>(ifs.tellg());
  ifs.seekg(0, std::ios::beg);

  auto &prop = tensor->properties;
  auto &vs = prop.validShape;
  size_t valid_elems = 1;
  for (int32_t i = 0; i < vs.numDimensions; ++i) {
    valid_elems *= static_cast<size_t>(vs.dimensionSize[i]);
  }
  const size_t expected_bytes = valid_elems * sizeof(float);
  if (file_bytes != expected_bytes) {
    VLOG(EXAMPLE_SYSTEM) << "Float bin size mismatch for " << bin_path
                         << ": file=" << file_bytes
                         << " expected=" << expected_bytes;
    return -1;
  }

  std::vector<float> data(valid_elems);
  ifs.read(reinterpret_cast<char *>(data.data()),
           static_cast<std::streamsize>(file_bytes));
  if (!ifs) {
    VLOG(EXAMPLE_SYSTEM) << "Read float bin failed: " << bin_path;
    return -1;
  }

  const auto t0 = Stopwatch::CurrentTs();
  memset(tensor->sysMem.virAddr, 0,
         static_cast<size_t>(prop.alignedByteSize));
  std::vector<uint32_t> dim(vs.numDimensions);
  std::vector<int64_t> stride(vs.numDimensions);
  for (int32_t i = 0; i < vs.numDimensions; ++i) {
    dim[i] = static_cast<uint32_t>(vs.dimensionSize[i]);
    stride[i] = prop.stride[i];
  }
  add_padding(tensor->sysMem.virAddr, data.data(), dim.size(), dim.data(),
              stride.data(), sizeof(float));
  hbUCPMemFlush(&tensor->sysMem, HB_SYS_MEM_CACHE_CLEAN);
  if (preprocess_us != nullptr) {
    *preprocess_us += Stopwatch::CurrentTs() - t0;
  }
  return 0;
}

int32_t QATBevFusionOcc4dPreProcessMethod::LoadAuxiliaryInput(
    hbDNNTensor *tensor, const std::string &bin_path, uint64_t *preprocess_us) {
  if (TryLoadExactAlignedBin(tensor, bin_path)) {
    return 0;
  }

  if (tensor->properties.tensorType == HB_DNN_TENSOR_TYPE_F32) {
    return LoadFloatInputFromBin(tensor, bin_path, preprocess_us);
  }

  // For quantized types (S8/S16/U8/U16), try loading bin and padding
  {
    auto &prop = tensor->properties;
    auto &vs = prop.validShape;
    size_t valid_elems = 1;
    for (int32_t i = 0; i < vs.numDimensions; ++i) {
      valid_elems *= static_cast<size_t>(vs.dimensionSize[i]);
    }

    // Determine element size for the quantized type
    size_t quant_elem_size = 0;
    if (prop.tensorType == HB_DNN_TENSOR_TYPE_S16) {
      quant_elem_size = sizeof(int16_t);
    } else if (prop.tensorType == HB_DNN_TENSOR_TYPE_S8) {
      quant_elem_size = sizeof(int8_t);
    }

    std::ifstream ifs(bin_path.c_str(), std::ios::in | std::ios::binary);
    if (ifs) {
      ifs.seekg(0, std::ios::end);
      size_t file_bytes = static_cast<size_t>(ifs.tellg());
      ifs.seekg(0, std::ios::beg);

      // Case 1: file is already quantized (matches valid_elems * quant_elem_size)
      if (quant_elem_size > 0 && file_bytes == valid_elems * quant_elem_size) {
        std::vector<char> raw(file_bytes);
        ifs.read(raw.data(), static_cast<std::streamsize>(file_bytes));
        if (ifs) {
          const auto t0 = Stopwatch::CurrentTs();
          memset(tensor->sysMem.virAddr, 0,
                 static_cast<size_t>(prop.alignedByteSize));
          std::vector<uint32_t> dim(vs.numDimensions);
          std::vector<int64_t> stride(vs.numDimensions);
          for (int32_t i = 0; i < vs.numDimensions; ++i) {
            dim[i] = static_cast<uint32_t>(vs.dimensionSize[i]);
            stride[i] = prop.stride[i];
          }
          add_padding(tensor->sysMem.virAddr, raw.data(), dim.size(),
                      dim.data(), stride.data(), quant_elem_size);
          hbUCPMemFlush(&tensor->sysMem, HB_SYS_MEM_CACHE_CLEAN);
          if (preprocess_us != nullptr) {
            *preprocess_us += Stopwatch::CurrentTs() - t0;
          }
          return 0;
        }
        // reset stream on failure
        ifs.clear();
        ifs.seekg(0, std::ios::beg);
      }

      // Case 2: file is float32 source, needs quantization
      size_t file_floats = file_bytes / sizeof(float);
      if (file_bytes % sizeof(float) == 0 && file_floats == valid_elems) {
        std::vector<float> fdata(valid_elems);
        ifs.read(reinterpret_cast<char *>(fdata.data()),
                 static_cast<std::streamsize>(file_bytes));
        if (ifs) {
          const auto t0 = Stopwatch::CurrentTs();
          memset(tensor->sysMem.virAddr, 0,
                 static_cast<size_t>(prop.alignedByteSize));

          float *scale_data = prop.scale.scaleData;
          float scale = (scale_data != nullptr) ? scale_data[0] : 1.0f;

          if (prop.tensorType == HB_DNN_TENSOR_TYPE_S16) {
            // Quantize float -> int16 with padding via stride
            std::vector<int16_t> qdata(valid_elems);
            for (size_t j = 0; j < valid_elems; ++j) {
              float q = std::round(fdata[j] / scale);
              q = std::max(std::min(q, 32767.0f), -32768.0f);
              qdata[j] = static_cast<int16_t>(q);
            }
            std::vector<uint32_t> dim(vs.numDimensions);
            std::vector<int64_t> stride(vs.numDimensions);
            for (int32_t i = 0; i < vs.numDimensions; ++i) {
              dim[i] = static_cast<uint32_t>(vs.dimensionSize[i]);
              stride[i] = prop.stride[i];
            }
            add_padding(tensor->sysMem.virAddr, qdata.data(), dim.size(),
                        dim.data(), stride.data(), sizeof(int16_t));
          } else if (prop.tensorType == HB_DNN_TENSOR_TYPE_S8) {
            std::vector<int8_t> qdata(valid_elems);
            for (size_t j = 0; j < valid_elems; ++j) {
              float q = std::round(fdata[j] / scale);
              q = std::max(std::min(q, 127.0f), -128.0f);
              qdata[j] = static_cast<int8_t>(q);
            }
            std::vector<uint32_t> dim(vs.numDimensions);
            std::vector<int64_t> stride(vs.numDimensions);
            for (int32_t i = 0; i < vs.numDimensions; ++i) {
              dim[i] = static_cast<uint32_t>(vs.dimensionSize[i]);
              stride[i] = prop.stride[i];
            }
            add_padding(tensor->sysMem.virAddr, qdata.data(), dim.size(),
                        dim.data(), stride.data(), sizeof(int8_t));
          } else {
            VLOG(EXAMPLE_SYSTEM) << "Unsupported quant tensor type: "
                                 << prop.tensorType;
            return -1;
          }

          hbUCPMemFlush(&tensor->sysMem, HB_SYS_MEM_CACHE_CLEAN);
          if (preprocess_us != nullptr) {
            *preprocess_us += Stopwatch::CurrentTs() - t0;
          }
          return 0;
        }
      }
    }
  }

  // Fallback: prepare_batch_tensor_* allocate their own memory
  release_tensor(tensor);
  std::string mutable_path = bin_path;
  if (prepare_batch_tensor_and_quanti(tensor, mutable_path) == 0) {
    return 0;
  }
  release_tensor(tensor);
  if (prepare_batch_tensor(tensor, mutable_path) == 0) {
    return 0;
  }
  return -1;
}

int32_t QATBevFusionOcc4dPreProcessMethod::LoadPointsInput(
    hbDNNTensor *tensor, const std::string &bin_path, uint64_t *preprocess_us) {
  if (TryLoadExactAlignedBin(tensor, bin_path)) {
    return 0;
  }

  std::ifstream ifs(bin_path.c_str(), std::ios::in | std::ios::binary);
  if (!ifs) {
    VLOG(EXAMPLE_SYSTEM) << "Open lidar bin failed: " << bin_path;
    return -1;
  }
  ifs.seekg(0, std::ios::end);
  const size_t file_bytes = static_cast<size_t>(ifs.tellg());
  ifs.seekg(0, std::ios::beg);
  if (file_bytes == 0 || file_bytes % sizeof(float) != 0) {
    VLOG(EXAMPLE_SYSTEM) << "Invalid lidar bin size: " << bin_path;
    return -1;
  }

  const size_t num_floats = file_bytes / sizeof(float);
  if (num_floats % static_cast<size_t>(point_dim_) != 0) {
    VLOG(EXAMPLE_SYSTEM) << "Lidar bin float count not divisible by point_dim="
                         << point_dim_ << ": " << bin_path;
    return -1;
  }

  const int32_t src_points =
      static_cast<int32_t>(num_floats / static_cast<size_t>(point_dim_));
  std::vector<float> src(num_floats);
  ifs.read(reinterpret_cast<char *>(src.data()),
           static_cast<std::streamsize>(file_bytes));
  if (!ifs) {
    VLOG(EXAMPLE_SYSTEM) << "Read lidar bin failed: " << bin_path;
    return -1;
  }

  auto &prop = tensor->properties;
  const int32_t cap_points = prop.validShape.dimensionSize[0];
  const int32_t out_dim = prop.validShape.dimensionSize[1];
  const int32_t row_stride =
      static_cast<int32_t>(prop.stride[0] / prop.stride[1]);
  float *dst = reinterpret_cast<float *>(tensor->sysMem.virAddr);
  const auto t0 = Stopwatch::CurrentTs();
  memset(dst, 0, static_cast<size_t>(prop.alignedByteSize));

  const bool filter_range = point_cloud_range_.size() >= 6;
  int32_t written = 0;
  for (int32_t i = 0; i < src_points && written < cap_points; ++i) {
    const float x = src[static_cast<size_t>(i) * point_dim_ + 0];
    const float y = src[static_cast<size_t>(i) * point_dim_ + 1];
    const float z = src[static_cast<size_t>(i) * point_dim_ + 2];
    if (filter_range) {
      if (x < point_cloud_range_[0] || x >= point_cloud_range_[3] ||
          y < point_cloud_range_[1] || y >= point_cloud_range_[4] ||
          z < point_cloud_range_[2] || z >= point_cloud_range_[5]) {
        continue;
      }
    }
    const int32_t copy_dim = std::min(point_dim_, out_dim);
    for (int32_t d = 0; d < copy_dim; ++d) {
      dst[written * row_stride + d] =
          src[static_cast<size_t>(i) * point_dim_ + d];
    }
    ++written;
  }

  hbUCPMemFlush(&tensor->sysMem, HB_SYS_MEM_CACHE_CLEAN);
  if (preprocess_us != nullptr) {
    *preprocess_us += Stopwatch::CurrentTs() - t0;
  }
  VLOG(EXAMPLE_DEBUG) << "Loaded " << written << " / " << cap_points
                      << " points from " << bin_path;
  return 0;
}

int32_t QATBevFusionOcc4dPreProcessMethod::LoadPrevBevInput(
    hbDNNTensor *tensor, const std::string &bin_path, uint64_t *preprocess_us) {
  if (TryLoadExactAlignedBin(tensor, bin_path)) {
    return 0;
  }

  std::ifstream ifs(bin_path.c_str(), std::ios::in | std::ios::binary);
  if (!ifs) {
    VLOG(EXAMPLE_SYSTEM) << "Open prev_bev bin failed: " << bin_path;
    return -1;
  }
  ifs.seekg(0, std::ios::end);
  const size_t file_bytes = static_cast<size_t>(ifs.tellg());
  ifs.seekg(0, std::ios::beg);

  auto &prop = tensor->properties;
  if (prop.tensorType != HB_DNN_TENSOR_TYPE_S8) {
    VLOG(EXAMPLE_SYSTEM) << "prev_bev input tensor must be INT8, got type "
                         << prop.tensorType;
    return -1;
  }

  auto &vs = prop.validShape;
  size_t valid_elems = 1;
  for (int32_t i = 0; i < vs.numDimensions; ++i) {
    valid_elems *= static_cast<size_t>(vs.dimensionSize[i]);
  }

  float dst_scale = 1.0f;
  if (prop.scale.scaleData != nullptr && prop.scale.scaleLen > 0) {
    dst_scale = prop.scale.scaleData[0];
  }
  if (dst_scale == 0.f) {
    dst_scale = 1.0f;
  }

  const auto t0 = Stopwatch::CurrentTs();
  memset(tensor->sysMem.virAddr, 0, static_cast<size_t>(prop.alignedByteSize));
  std::vector<uint32_t> dim(vs.numDimensions);
  std::vector<int64_t> stride(vs.numDimensions);
  for (int32_t i = 0; i < vs.numDimensions; ++i) {
    dim[i] = static_cast<uint32_t>(vs.dimensionSize[i]);
    stride[i] = prop.stride[i];
  }

  if (file_bytes == valid_elems * sizeof(int8_t)) {
    std::vector<int8_t> qdata(valid_elems);
    ifs.read(reinterpret_cast<char *>(qdata.data()),
             static_cast<std::streamsize>(file_bytes));
    if (!ifs) {
      VLOG(EXAMPLE_SYSTEM) << "Read prev_bev INT8 bin failed: " << bin_path;
      return -1;
    }
    add_padding(tensor->sysMem.virAddr, qdata.data(), dim.size(), dim.data(),
                stride.data(), sizeof(int8_t));
  } else if (file_bytes == valid_elems * sizeof(int16_t)) {
    if (prev_bev_src_scale_ == 0.f) {
      VLOG(EXAMPLE_SYSTEM)
          << "prev_bev.bin is INT16 (output_1 format) but prev_bev_src_scale "
             "is not set in preprocess config: "
          << bin_path;
      return -1;
    }
    std::vector<int16_t> src(valid_elems);
    ifs.read(reinterpret_cast<char *>(src.data()),
             static_cast<std::streamsize>(file_bytes));
    if (!ifs) {
      VLOG(EXAMPLE_SYSTEM) << "Read prev_bev INT16 bin failed: " << bin_path;
      return -1;
    }
    std::vector<int8_t> qdata(valid_elems);
    for (size_t i = 0; i < valid_elems; ++i) {
      float dequant = static_cast<float>(src[i]) * prev_bev_src_scale_;
      float q = std::round(dequant / dst_scale);
      q = std::max(std::min(q, 127.0f), -128.0f);
      qdata[i] = static_cast<int8_t>(q);
    }
    add_padding(tensor->sysMem.virAddr, qdata.data(), dim.size(), dim.data(),
                stride.data(), sizeof(int8_t));
    VLOG(EXAMPLE_DEBUG) << "Converted prev_bev INT16 -> INT8 from " << bin_path
                        << " (src_scale=" << prev_bev_src_scale_
                        << ", dst_scale=" << dst_scale << ")";
  } else if (file_bytes == valid_elems * sizeof(float)) {
    std::vector<float> fdata(valid_elems);
    ifs.read(reinterpret_cast<char *>(fdata.data()),
             static_cast<std::streamsize>(file_bytes));
    if (!ifs) {
      VLOG(EXAMPLE_SYSTEM) << "Read prev_bev float bin failed: " << bin_path;
      return -1;
    }
    std::vector<int8_t> qdata(valid_elems);
    for (size_t i = 0; i < valid_elems; ++i) {
      float q = std::round(fdata[i] / dst_scale);
      q = std::max(std::min(q, 127.0f), -128.0f);
      qdata[i] = static_cast<int8_t>(q);
    }
    add_padding(tensor->sysMem.virAddr, qdata.data(), dim.size(), dim.data(),
                stride.data(), sizeof(int8_t));
  } else {
    VLOG(EXAMPLE_SYSTEM) << "prev_bev bin size mismatch for " << bin_path
                         << ": file=" << file_bytes
                         << " expected one of "
                         << valid_elems * sizeof(int8_t) << "(INT8), "
                         << valid_elems * sizeof(int16_t) << "(INT16), "
                         << valid_elems * sizeof(float) << "(FLOAT32)";
    return -1;
  }

  hbUCPMemFlush(&tensor->sysMem, HB_SYS_MEM_CACHE_CLEAN);
  if (preprocess_us != nullptr) {
    *preprocess_us += Stopwatch::CurrentTs() - t0;
  }
  return 0;
}

int32_t QATBevFusionOcc4dPreProcessMethod::InitFromJsonString(
    const std::string &config) {
  VLOG(EXAMPLE_DEBUG) << "QATBevFusionOcc4dPreProcessMethod Json string:"
                      << config.data();

  rapidjson::Document document;
  document.Parse(config.data());
  if (document.HasParseError()) {
    VLOG(EXAMPLE_SYSTEM) << "Parsing config file failed";
    return -1;
  }

  if (document.HasMember("auxiliary_bin_names")) {
    auto arr = document["auxiliary_bin_names"].GetArray();
    auxiliary_bin_names_.clear();
    for (rapidjson::SizeType i = 0; i < arr.Size(); ++i) {
      auxiliary_bin_names_.push_back(arr[i].GetString());
    }
  }

  if (document.HasMember("lidar_bin_name")) {
    lidar_bin_name_ = document["lidar_bin_name"].GetString();
  }

  if (document.HasMember("prev_bev_bin_name")) {
    prev_bev_bin_name_ = document["prev_bev_bin_name"].GetString();
  }

  if (document.HasMember("prev_bev_src_scale")) {
    prev_bev_src_scale_ = document["prev_bev_src_scale"].GetFloat();
  }

  if (document.HasMember("point_cloud_range")) {
    auto arr = document["point_cloud_range"].GetArray();
    point_cloud_range_.resize(arr.Size());
    for (rapidjson::SizeType i = 0; i < arr.Size(); ++i) {
      point_cloud_range_[i] = arr[i].GetFloat();
    }
  }

  if (document.HasMember("max_num_points")) {
    max_num_points_ = document["max_num_points"].GetInt();
  }

  if (document.HasMember("point_dim")) {
    point_dim_ = document["point_dim"].GetInt();
  }

  if (static_cast<int>(auxiliary_bin_names_.size()) != kExpectedCameraAuxInputs) {
    VLOG(EXAMPLE_SYSTEM)
        << "auxiliary_bin_names must have " << kExpectedCameraAuxInputs
        << " entries (imgs, uv, zcam), got "
        << auxiliary_bin_names_.size();
    return -1;
  }

  return 0;
}

int32_t QATBevFusionOcc4dPreProcessMethod::DoProcess(std::string path,
                                                   int32_t input_count,
                                                   ImageTensor *image_tensor) {
  if (input_count != kExpectedInputCount) {
    VLOG(EXAMPLE_SYSTEM) << "BEVFusion OCC expects " << kExpectedInputCount
                         << " inputs, got " << input_count;
    return -1;
  }

  if (!IsDirectory(path)) {
    VLOG(EXAMPLE_SYSTEM)
        << "BEVFusion OCC preprocess expects a sample directory path: " << path;
    return -1;
  }

  uint64_t preprocess_us = 0;
  auto &tensors = image_tensor->tensors;
  tensors.resize(input_count);

  {
    const auto t0 = Stopwatch::CurrentTs();
    for (int i = 0; i < input_count; ++i) {
      hbDNNGetInputTensorProperties(&(tensors[i].properties), dnn_handle_, i);
      hbUCPMallocCached(&(tensors[i].sysMem),
                        tensors[i].properties.alignedByteSize, 0);
    }
    preprocess_us += Stopwatch::CurrentTs() - t0;
  }

  for (int i = 0; i < kExpectedCameraAuxInputs; ++i) {
    const std::string bin_path = JoinPath(path, auxiliary_bin_names_[i]);
    if (LoadAuxiliaryInput(&tensors[i], bin_path, &preprocess_us) != 0) {
      VLOG(EXAMPLE_SYSTEM) << "Load camera/aux input " << i
                           << " failed: " << bin_path;
      return -1;
    }
  }

  const std::string lidar_path = JoinPath(path, lidar_bin_name_);
  if (LoadPointsInput(&tensors[kPointsInputIndex], lidar_path, &preprocess_us) !=
      0) {
    VLOG(EXAMPLE_SYSTEM) << "Load lidar points failed: " << lidar_path;
    return -1;
  }

  const std::string prev_bev_path = JoinPath(path, prev_bev_bin_name_);
  if (LoadPrevBevInput(&tensors[kPrevBevInputIndex], prev_bev_path,
                       &preprocess_us) != 0) {
    VLOG(EXAMPLE_SYSTEM) << "Load prev_bev failed: " << prev_bev_path;
    return -1;
  }

  {
    const auto t0 = Stopwatch::CurrentTs();
    flush_multi_tensor(tensors);
    preprocess_us += Stopwatch::CurrentTs() - t0;
  }

  image_tensor->pre_duration = preprocess_us;
  // VLOG(EXAMPLE_REPORT) << std::fixed << std::setprecision(3)
  //                      << "BevFusion OCC preprocess frame_id="
  //                      << image_tensor->frame_id
  //                      << " pre_ms=" << image_tensor->pre_duration / 1000.0;
  return 0;
}
