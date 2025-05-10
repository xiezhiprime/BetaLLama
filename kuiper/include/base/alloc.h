#ifndef KUIPER_INCLUDE_BASE_ALLOC_H_
#define KUIPER_INCLUDE_BASE_ALLOC_H_
#include <map>
#include <memory>
#include "base.h"
namespace base {
enum class MemcpyKind {
  kMemcpyCPU2CPU = 0,
  kMemcpyCPU2CUDA = 1,
  kMemcpyCUDA2CPU = 2,
  kMemcpyCUDA2CUDA = 3,
};

class DeviceAllocator {
 public:
  explicit DeviceAllocator(DeviceType device_type) : device_type_(device_type) {}

  virtual DeviceType device_type() const { return device_type_; }

  virtual void release(void* ptr) const = 0;

  // 声明一个名为 allocate 的纯虚函数, 
  // 该函数接收一个 size_t 参数并返回 void* 指针, 该函数不会修改对象的状态（const）
  // 继承 DeviceAllocator 的类（如 CPUDeviceAllocator 和 CUDADeviceAllocator）必须提供这个函数的实现
  virtual void* allocate(size_t byte_size) const = 0;

  /*
    params:
      src_ptr：源数据指针
      dest_ptr：目标数据指针
      byte_size：要复制的字节数
      memcpy_kind：复制类型（CPU到CPU、CPU到CUDA、CUDA到CPU或CUDA到CUDA）
      stream：可选的CUDA流指针，用于异步操作
      need_sync：是否需要在复制后同步（等待操作完成）
  */
  virtual void memcpy(const void* src_ptr, void* dest_ptr, size_t byte_size,
                      MemcpyKind memcpy_kind = MemcpyKind::kMemcpyCPU2CPU, void* stream = nullptr,
                      bool need_sync = false) const;

  virtual void memset_zero(void* ptr, size_t byte_size, void* stream, bool need_sync = false);

 private:
  // device_type_ 表示该资源申请类所属的设备类型
  DeviceType device_type_ = DeviceType::kDeviceUnknown;
};

class CPUDeviceAllocator : public DeviceAllocator {
 public:
  explicit CPUDeviceAllocator();

  void* allocate(size_t byte_size) const override;

  void release(void* ptr) const override;
};

struct CudaMemoryBuffer {
  void* data;
  size_t byte_size;
  bool busy;

  CudaMemoryBuffer() = default;

  CudaMemoryBuffer(void* data, size_t byte_size, bool busy)
      : data(data), byte_size(byte_size), busy(busy) {}
};

class CUDADeviceAllocator : public DeviceAllocator {
 public:
  explicit CUDADeviceAllocator();

  void* allocate(size_t byte_size) const override;

  void release(void* ptr) const override;

 private:
  mutable std::map<int, size_t> no_busy_cnt_;
  mutable std::map<int, std::vector<CudaMemoryBuffer>> big_buffers_map_;
  mutable std::map<int, std::vector<CudaMemoryBuffer>> cuda_buffers_map_;
};

class CPUDeviceAllocatorFactory {
 public:
  static std::shared_ptr<CPUDeviceAllocator> get_instance() {
    if (instance == nullptr) {
      instance = std::make_shared<CPUDeviceAllocator>();
    }
    return instance;
  }

 private:
  static std::shared_ptr<CPUDeviceAllocator> instance;
};

class CUDADeviceAllocatorFactory {
 public:
  static std::shared_ptr<CUDADeviceAllocator> get_instance() {
    if (instance == nullptr) {
      instance = std::make_shared<CUDADeviceAllocator>();
    }
    return instance;
  }

 private:
  static std::shared_ptr<CUDADeviceAllocator> instance;
};
}  // namespace base
#endif  // KUIPER_INCLUDE_BASE_ALLOC_H_