#include <cuda_runtime_api.h>
#include "base/alloc.h"
namespace base {

CUDADeviceAllocator::CUDADeviceAllocator() : DeviceAllocator(DeviceType::kDeviceCUDA) {}

void* CUDADeviceAllocator::allocate(size_t byte_size) const {
  // 1. 初始化与设备标识 : 获取当前CUDA设备ID，为后续在特定设备上分配内存做准备。
  int id = -1;
  cudaError_t state = cudaGetDevice(&id);
  CHECK(state == cudaSuccess);
  // 2. 大内存分配逻辑（>1MB）
  if (byte_size > 1024 * 1024) {
    auto& big_buffers = big_buffers_map_[id];
    // 查找最适合的缓冲区：
    //  检查每个缓冲区是否满足：尺寸足够、未被占用、浪费空间不超过1MB
    //  选择满足条件的最小缓冲区，减少内存碎片
    int sel_id = -1;
    for (int i = 0; i < big_buffers.size(); i++) {
      if (big_buffers[i].byte_size >= byte_size && !big_buffers[i].busy &&
          big_buffers[i].byte_size - byte_size < 1 * 1024 * 1024) {
        if (sel_id == -1 || big_buffers[sel_id].byte_size > big_buffers[i].byte_size) {
          sel_id = i;
        }
      }
    }
    // 复用现有缓冲区：若找到合适的，则标记为忙碌并返回
    if (sel_id != -1) {
      big_buffers[sel_id].busy = true;
      return big_buffers[sel_id].data;
    }
    // 分配新内存：若无法复用，调用cudaMalloc分配新内存，并加入内存池
    void* ptr = nullptr;
    state = cudaMalloc(&ptr, byte_size);
    // 4. 错误处理 : 分配失败时，函数提供详细错误信息，指明可能是设备内存不足导致。
    if (cudaSuccess != state) {
      char buf[256];
      snprintf(buf, 256,
               "Error: CUDA error when allocating %lu MB memory! maybe there's no enough memory "
               "left on  device.",
               byte_size >> 20);
      LOG(ERROR) << buf;
      return nullptr;
    }
    big_buffers.emplace_back(ptr, byte_size, true);
    return ptr;
  }

  // 3. 小内存分配逻辑（≤1MB）
  // 寻找可用缓冲区： 
  //    遍历小内存池，找到第一个尺寸足够且未被占用的缓冲区
  //    标记为忙碌并更新未使用内存计数器
  auto& cuda_buffers = cuda_buffers_map_[id];
  for (int i = 0; i < cuda_buffers.size(); i++) {
    if (cuda_buffers[i].byte_size >= byte_size && !cuda_buffers[i].busy) {
      cuda_buffers[i].busy = true;
      no_busy_cnt_[id] -= cuda_buffers[i].byte_size;
      return cuda_buffers[i].data;
    }
  }
  // 无可复用则新分配：与大内存类似，但记录到小内存池中
  void* ptr = nullptr;
  state = cudaMalloc(&ptr, byte_size);
  // 4. 错误处理 : 分配失败时，函数提供详细错误信息，指明可能是设备内存不足导致。
  if (cudaSuccess != state) {
    char buf[256];
    snprintf(buf, 256,
             "Error: CUDA error when allocating %lu MB memory! maybe there's no enough memory "
             "left on  device.",
             byte_size >> 20);
    LOG(ERROR) << buf;
    return nullptr;
  }
  cuda_buffers.emplace_back(ptr, byte_size, true);
  return ptr;
}

void CUDADeviceAllocator::release(void* ptr) const {
  if (!ptr) {
    return;
  }
  if (cuda_buffers_map_.empty()) {
    return;
  }
  cudaError_t state = cudaSuccess;
  for (auto& it : cuda_buffers_map_) {
    if (no_busy_cnt_[it.first] > 1024 * 1024 * 1024) {
      auto& cuda_buffers = it.second;
      std::vector<CudaMemoryBuffer> temp;
      for (int i = 0; i < cuda_buffers.size(); i++) {
        if (!cuda_buffers[i].busy) {
          state = cudaSetDevice(it.first);
          state = cudaFree(cuda_buffers[i].data);
          CHECK(state == cudaSuccess)
              << "Error: CUDA error when release memory on device " << it.first;
        } else {
          temp.push_back(cuda_buffers[i]);
        }
      }
      cuda_buffers.clear();
      it.second = temp;
      no_busy_cnt_[it.first] = 0;
    }
  }

  for (auto& it : cuda_buffers_map_) {
    auto& cuda_buffers = it.second;
    for (int i = 0; i < cuda_buffers.size(); i++) {
      if (cuda_buffers[i].data == ptr) {
        no_busy_cnt_[it.first] += cuda_buffers[i].byte_size;
        cuda_buffers[i].busy = false;
        return;
      }
    }
    auto& big_buffers = big_buffers_map_[it.first];
    for (int i = 0; i < big_buffers.size(); i++) {
      if (big_buffers[i].data == ptr) {
        big_buffers[i].busy = false;
        return;
      }
    }
  }
  state = cudaFree(ptr);
  CHECK(state == cudaSuccess) << "Error: CUDA error when release memory on device";
}
std::shared_ptr<CUDADeviceAllocator> CUDADeviceAllocatorFactory::instance = nullptr;

}  // namespace base