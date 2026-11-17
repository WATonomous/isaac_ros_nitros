// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// SPDX-License-Identifier: Apache-2.0
#include <cuda_runtime.h>

#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "isaac_ros_common/cuda_stream.hpp"
#include "isaac_ros_nitros/types/cuda_stream_pool.hpp"
#include "isaac_ros_nitros_point_cloud_type/nitros_point_cloud.hpp"

#ifdef NITROS_POINT_CLOUD_GXF_COMPAT_MODE
#include "isaac_ros_nitros/types/type_adapter_nitros_context.hpp"
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wpedantic"
#include "gxf/core/entity.hpp"
#include "gxf/core/gxf.h"
#include "messages/point_cloud_message.hpp"
#pragma GCC diagnostic pop
#endif

#include "rclcpp/rclcpp.hpp"

void rclcpp::TypeAdapter<
  nvidia::isaac_ros::nitros::NitrosPointCloud,
  sensor_msgs::msg::PointCloud2>::convert_to_ros_message(
  const custom_type & source, ros_message_type & destination)
{
  nvidia::isaac_ros::nitros::nvtxRangePushWrapper(
    "NitrosPointCloud::convert_to_ros_message",
    nvidia::isaac_ros::nitros::CLR_PURPLE);

  RCLCPP_DEBUG(
    rclcpp::get_logger("NitrosPointCloud"),
    "[convert_to_ros_message] Conversion started for handle=%ld", source.handle);

#ifdef NITROS_POINT_CLOUD_GXF_COMPAT_MODE
  if (source.handle >= 0) {
    auto context = nvidia::isaac_ros::nitros::GetTypeAdapterNitrosContext().getContext();
    auto entity = nvidia::gxf::Entity::Shared(context, source.handle);
    if (!entity) {
      throw std::runtime_error("NitrosPointCloud GXF compat: Failed to get GXF entity");
    }
    auto maybe_parts = nvidia::isaac_ros::messages::GetPointCloudMessage(entity.value());
    if (!maybe_parts) {
      std::stringstream error_msg;
      error_msg << "[convert_to_ros_message] Failed to get point cloud message: "
                << GxfResultStr(maybe_parts.error());
      RCLCPP_ERROR(rclcpp::get_logger("NitrosPointCloud"), error_msg.str().c_str());
      throw std::runtime_error(error_msg.str().c_str());
    }
    auto parts = maybe_parts.value();
    destination.height = 1;
    destination.width = static_cast<uint32_t>(parts.points->shape().dimension(0));
    destination.point_step = parts.info->use_color ? 16u : 12u;
    destination.row_step = destination.point_step * destination.width;
    destination.is_bigendian = parts.info->is_bigendian;
    destination.is_dense = false;
    sensor_msgs::PointCloud2Modifier pc2_modifier(destination);
    if (parts.info->use_color) {
      pc2_modifier.setPointCloud2Fields(
        4,
        "x", 1, sensor_msgs::msg::PointField::FLOAT32,
        "y", 1, sensor_msgs::msg::PointField::FLOAT32,
        "z", 1, sensor_msgs::msg::PointField::FLOAT32,
        "rgb", 1, sensor_msgs::msg::PointField::FLOAT32);
    } else {
      pc2_modifier.setPointCloud2Fields(
        3,
        "x", 1, sensor_msgs::msg::PointField::FLOAT32,
        "y", 1, sensor_msgs::msg::PointField::FLOAT32,
        "z", 1, sensor_msgs::msg::PointField::FLOAT32);
    }
    destination.data.resize(destination.row_step * destination.height);
    auto maybe_src = parts.points->data<float>();
    if (!maybe_src) {
      throw std::runtime_error("[convert_to_ros_message] Tensor returned no data pointer");
    }
    const size_t copy_bytes = destination.row_step * destination.height;
    const cudaError_t err = cudaMemcpy(
      destination.data.data(), *maybe_src, copy_bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
      std::stringstream error_msg;
      error_msg << "[convert_to_ros_message] cudaMemcpy failed: "
                << cudaGetErrorName(err) << " (" << cudaGetErrorString(err) << ")";
      RCLCPP_ERROR(rclcpp::get_logger("NitrosPointCloud"), error_msg.str().c_str());
      throw std::runtime_error(error_msg.str().c_str());
    }
    destination.header.stamp.sec = source.timestamp_sec;
    destination.header.stamp.nanosec = source.timestamp_nsec;
    destination.header.frame_id = source.frame_id;
    nvidia::isaac_ros::nitros::nvtxRangePopWrapper();
    return;
  }
#endif

  destination.height = source.height;
  destination.width = source.width;
  destination.point_step = source.point_step;
  destination.row_step = source.row_step;
  destination.is_bigendian = source.is_bigendian;
  destination.is_dense = source.is_dense;
  destination.fields = source.fields;
  const size_t bytes = nvidia::isaac_ros::nitros::ValidatePointCloudLayout(
      source.width, source.height, source.point_step, source.row_step, source.fields);
  if (bytes > source.get_buffer_size())
  {
    throw std::invalid_argument("PointCloud2 metadata exceeds GPU buffer size");
  }
  destination.data.resize(bytes);
  if (bytes)
  {
    auto stream_handle = nvidia::isaac_ros::nitros::CudaStreamPool::instance().get_stream_handle();
    auto stream = stream_handle.get();
    auto read_handle = source.get_read_handle(stream);
    CHECK_CUDA_ERROR(cudaMemcpyAsync(destination.data.data(), read_handle.get_ptr(), bytes,
                                     cudaMemcpyDeviceToHost, stream),
                     "PointCloud2 download failed");
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream), "PointCloud2 download synchronization failed");
  }

  // Populate timestamp information back into ROS header
  destination.header.stamp.sec = source.timestamp_sec;
  destination.header.stamp.nanosec = source.timestamp_nsec;
  destination.header.frame_id = source.frame_id;

  RCLCPP_DEBUG(
    rclcpp::get_logger("NitrosPointCloud"),
    "[convert_to_ros_message] Conversion completed");

  nvidia::isaac_ros::nitros::nvtxRangePopWrapper();
}

void rclcpp::TypeAdapter<
  nvidia::isaac_ros::nitros::NitrosPointCloud,
  sensor_msgs::msg::PointCloud2>::convert_to_custom(
  const ros_message_type & source,
  custom_type & destination)
{
  nvidia::isaac_ros::nitros::nvtxRangePushWrapper(
    "NitrosPointCloud::convert_to_custom",
    nvidia::isaac_ros::nitros::CLR_PURPLE);

  RCLCPP_DEBUG(
    rclcpp::get_logger("NitrosPointCloud"),
    "[convert_to_custom] Conversion started");

  using namespace nvidia::isaac_ros::nitros;
  const size_t total_bytes = ValidatePointCloudLayout(
      source.width, source.height, source.point_step, source.row_step, source.fields);
  if (source.data.size() != total_bytes)
  {
    throw std::invalid_argument("PointCloud2 data size does not match row_step * height");
  }

  NitrosPointCloud point_cloud;
  if (total_bytes)
  {
    auto stream_handle = CudaStreamPool::instance().get_stream_handle();
    auto stream = stream_handle.get();
    uint8_t *dptr = nullptr;
    CHECK_CUDA_ERROR(cudaMalloc(reinterpret_cast<void **>(&dptr), total_bytes),
                     "PointCloud2 allocation failed");
    // from_external takes ownership; a shared buffer survives copies through message_filters.
    try
    {
      {
        auto write_handle = point_cloud.from_external(
            dptr, total_bytes, source.width, source.height, source.point_step, source.row_step,
            source.is_bigendian, false, stream);
        dptr = nullptr;
        CHECK_CUDA_ERROR(cudaMemcpyAsync(write_handle.get_ptr(), source.data.data(), total_bytes,
                                         cudaMemcpyHostToDevice, stream),
                         "PointCloud2 upload failed");
      }
      // The ROS source may be destroyed on return: finish reading its host storage first.
      CHECK_CUDA_ERROR(cudaStreamSynchronize(stream), "PointCloud2 upload synchronization failed");
    }
    catch (...)
    {
      if (dptr)
      {
        cudaFree(dptr);
      }
      throw;
    }
  }
  point_cloud.width = source.width;
  point_cloud.height = source.height;
  point_cloud.point_step = source.point_step;
  point_cloud.row_step = source.row_step;
  point_cloud.is_bigendian = source.is_bigendian;
  point_cloud.is_dense = source.is_dense;
  point_cloud.fields = source.fields;
  point_cloud.use_color =
      std::any_of(source.fields.begin(), source.fields.end(),
                  [](const auto &field) { return field.name == "rgb" || field.name == "rgba"; });
  point_cloud.timestamp_sec = source.header.stamp.sec;
  point_cloud.timestamp_nsec = source.header.stamp.nanosec;
  point_cloud.frame_id = source.header.frame_id;
  destination = std::move(point_cloud);

  RCLCPP_DEBUG(
    rclcpp::get_logger("NitrosPointCloud"),
    "[convert_to_custom] Conversion completed");

  nvidia::isaac_ros::nitros::nvtxRangePopWrapper();
}
