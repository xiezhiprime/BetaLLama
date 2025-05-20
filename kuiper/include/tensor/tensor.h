#ifndef KUIPER_INCLUDE_TENSOR_TENSOR_H_
#define KUIPER_INCLUDE_TENSOR_TENSOR_H_
#include <driver_types.h>
#include <glog/logging.h>
#include <armadillo>
#include <memory>
#include <vector>
#include "base/base.h"
#include "base/buffer.h"
namespace tensor {

class Tensor {
 public:
  // 创建一个空张量，所有成员使用默认值，适用于需要后续再设置属性的情况（延迟初始化）
  explicit Tensor() = default;
  // 固定维度构造函数（1D到4D）
  explicit Tensor(base::DataType data_type, int32_t dim0, bool need_alloc = false,
                  std::shared_ptr<base::DeviceAllocator> alloc = nullptr, void* ptr = nullptr);
  // 固定维度构造函数（1D到4D）
  explicit Tensor(base::DataType data_type, int32_t dim0, int32_t dim1, bool need_alloc = false,
                  std::shared_ptr<base::DeviceAllocator> alloc = nullptr, void* ptr = nullptr);
  // 固定维度构造函数（1D到4D）
  explicit Tensor(base::DataType data_type, int32_t dim0, int32_t dim1, int32_t dim2,
                  bool need_alloc = false, std::shared_ptr<base::DeviceAllocator> alloc = nullptr,
                  void* ptr = nullptr);
  // 固定维度构造函数（1D到4D）
  explicit Tensor(base::DataType data_type, int32_t dim0, int32_t dim1, int32_t dim2, int32_t dim3,
                  bool need_alloc = false, std::shared_ptr<base::DeviceAllocator> alloc = nullptr,
                  void* ptr = nullptr);
  // 支持任意维度的张量创建，适用于运行时确定维度或高维张量，更灵活，但少了编译时的维度检查。
  explicit Tensor(base::DataType data_type, std::vector<int32_t> dims, bool need_alloc = false,
                  std::shared_ptr<base::DeviceAllocator> alloc = nullptr, void* ptr = nullptr);

  void to_cpu();

  void to_cuda(cudaStream_t stream = nullptr);

  bool is_empty() const;

  void init_buffer(std::shared_ptr<base::DeviceAllocator> alloc, base::DataType data_type,
                   bool need_alloc, void* ptr);

  template <typename T>
  T* ptr();

  template <typename T>
  const T* ptr() const;

  void reshape(const std::vector<int32_t>& dims);

  std::shared_ptr<base::Buffer> get_buffer() const;

  size_t size() const;

  size_t byte_size() const;

  int32_t dims_size() const;

  base::DataType data_type() const;

  int32_t get_dim(int32_t idx) const;

  const std::vector<int32_t>& dims() const;

  std::vector<size_t> strides() const;

  bool assign(std::shared_ptr<base::Buffer> buffer);

  void reset(base::DataType data_type, const std::vector<int32_t>& dims);

  void set_device_type(base::DeviceType device_type) const;

  base::DeviceType device_type() const;

  bool allocate(std::shared_ptr<base::DeviceAllocator> allocator, bool need_realloc = false);

  template <typename T>
  T* ptr(int64_t index);

  template <typename T>
  const T* ptr(int64_t index) const;

  template <typename T>
  T& index(int64_t offset);

  template <typename T>
  const T& index(int64_t offset) const;

  tensor::Tensor clone() const;

 private:
  // 张量中数据的个数，比如张量后期要存储3个数据，分别为{1，2，3}，那么size的大小就等于3。
  size_t size_ = 0;
  // 张量的维度，比如有一个二维张量，且维度分别是{2, 3}，那么dim_记录的值就是{2,3}
  std::vector<int32_t> dims_;
  // 用来管理用分配器申请到的内存资源，在Buffer中我们用到RAII和智能指针技术。
  std::shared_ptr<base::Buffer> buffer_;
  base::DataType data_type_ = base::DataType::kDataTypeUnknown;
};

template <typename T>
T& Tensor::index(int64_t offset) {
  CHECK_GE(offset, 0);
  CHECK_LT(offset, this->size());
  T& val = *(reinterpret_cast<T*>(buffer_->ptr()) + offset);
  return val;
}

template <typename T>
const T& Tensor::index(int64_t offset) const {
  CHECK_GE(offset, 0);
  CHECK_LT(offset, this->size());
  const T& val = *(reinterpret_cast<T*>(buffer_->ptr()) + offset);
  return val;
}

template <typename T>
const T* Tensor::ptr() const {
  if (!buffer_) {
    return nullptr;
  }
  return const_cast<const T*>(reinterpret_cast<T*>(buffer_->ptr()));
}

template <typename T>
T* Tensor::ptr() {
  if (!buffer_) {
    return nullptr;
  }
  return reinterpret_cast<T*>(buffer_->ptr());
}

template <typename T>
T* Tensor::ptr(int64_t index) {
  CHECK(buffer_ != nullptr && buffer_->ptr() != nullptr)
      << "The data area buffer of this tensor is empty or it points to a null pointer.";
  return const_cast<T*>(reinterpret_cast<const T*>(buffer_->ptr())) + index;
}

template <typename T>
const T* Tensor::ptr(int64_t index) const {
  CHECK(buffer_ != nullptr && buffer_->ptr() != nullptr)
      << "The data area buffer of this tensor is empty or it points to a null pointer.";
  return reinterpret_cast<const T*>(buffer_->ptr()) + index;
}
}  // namespace tensor
#endif  // KUIPER_INCLUDE_TENSOR_TENSOR_H_
