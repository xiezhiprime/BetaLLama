#include "base/alloc.h"
#include <cuda_runtime_api.h>
namespace base {
void DeviceAllocator::memcpy(const void* src_ptr, void* dest_ptr, size_t byte_size,
                             MemcpyKind memcpy_kind, void* stream, bool need_sync) const {
  // 1.输入验证： 检查源指针与目标指针不为空，检查要复制的字节数是否有效。
  CHECK_NE(src_ptr, nullptr);
  CHECK_NE(dest_ptr, nullptr);
  if (!byte_size) {
    return;
  }
  // 2.CUDA流处理：如果提供了流参数，则将其转换为CUDA流类型
  cudaStream_t stream_ = nullptr;
  if (stream) {
    stream_ = static_cast<CUstream_st*>(stream);
  }
  // 3.根据复制类型执行不同的操作
  if (memcpy_kind == MemcpyKind::kMemcpyCPU2CPU) {
    std::memcpy(dest_ptr, src_ptr, byte_size);
  } else if (memcpy_kind == MemcpyKind::kMemcpyCPU2CUDA) {
    // 根据是否有流，选择同步(cudaMemcpy)或异步(cudaMemcpyAsync)操作
    if (!stream_) {
      cudaMemcpy(dest_ptr, src_ptr, byte_size, cudaMemcpyHostToDevice);
    } else {
      cudaMemcpyAsync(dest_ptr, src_ptr, byte_size, cudaMemcpyHostToDevice, stream_);
    }
  } else if (memcpy_kind == MemcpyKind::kMemcpyCUDA2CPU) {
    // 根据是否有流，选择同步(cudaMemcpy)或异步(cudaMemcpyAsync)操作
    if (!stream_) {
      cudaMemcpy(dest_ptr, src_ptr, byte_size, cudaMemcpyDeviceToHost);
    } else {
      cudaMemcpyAsync(dest_ptr, src_ptr, byte_size, cudaMemcpyDeviceToHost, stream_);
    }
  } else if (memcpy_kind == MemcpyKind::kMemcpyCUDA2CUDA) {
    // 根据是否有流，选择同步(cudaMemcpy)或异步(cudaMemcpyAsync)操作
    if (!stream_) {
      cudaMemcpy(dest_ptr, src_ptr, byte_size, cudaMemcpyDeviceToDevice);
    } else {
      cudaMemcpyAsync(dest_ptr, src_ptr, byte_size, cudaMemcpyDeviceToDevice, stream_);
    }
  } else {
    LOG(FATAL) << "Unknown memcpy kind: " << int(memcpy_kind);
  }
  if (need_sync) {
    // 如果需要同步，调用cudaDeviceSynchronize()等待所有CUDA操作完成
    cudaDeviceSynchronize();
  }
}

void DeviceAllocator::memset_zero(void* ptr, size_t byte_size, void* stream,
                                  bool need_sync) {
  CHECK(device_type_ != base::DeviceType::kDeviceUnknown);
  if (device_type_ == base::DeviceType::kDeviceCPU) {
    std::memset(ptr, 0, byte_size);
  } else {
    if (stream) {
      cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
      cudaMemsetAsync(ptr, 0, byte_size, stream_);
    } else {
      cudaMemset(ptr, 0, byte_size);
    }
    if (need_sync) {
      cudaDeviceSynchronize();
    }
  }
}

}  // namespace base