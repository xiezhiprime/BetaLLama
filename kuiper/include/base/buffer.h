#ifndef KUIPER_INCLUDE_BASE_BUFFER_H_
#define KUIPER_INCLUDE_BASE_BUFFER_H_
#include <memory>
#include "base/alloc.h"
namespace base {
class Buffer : public NoCopyable, std::enable_shared_from_this<Buffer> {
 private:
  // 这块内存的大小，以字节数作为单位。
  size_t byte_size_ = 0;
  // 这块内存的地址，主要有两种来源
  //  一种是外部直接赋值得到的， Buffer不需要对它进行管理，和它的关系是借用，不负责它的生命周期管理，这种情况下对应下方use_external的值置为true。
  //  另外一种是需要Buffer对这块内存进行管理的，所以use_external值为false，表示需要对它的生命周期进行管理，也就是没人使用该Buffer的时候会自动将ptr_指向的地址用对应类型的Allocator完成释放。
  void* ptr_ = nullptr;
  bool use_external_ = false;
  // 表示Buffer中内存资源所属的设备类型
  DeviceType device_type_ = DeviceType::kDeviceUnknown;
  // Buffer对应设备类型的内存分配器，负责资源的释放、申请以及拷贝等，
  // 既可以是cpu allocator 也可以是cuda allocator.
  std::shared_ptr<DeviceAllocator> allocator_;

 public:
  /*
    explicit 用于构造函数声明，表明该构造函数不能用于隐式类型转换。
    对于不接受参数的默认构造函数，使用 explicit 在技术上不是必需的（因为没有可能的隐式转换），但这样做可以：
      表明程序员的明确意图
      保持与类中其他构造函数声明风格的一致性
      防止未来可能的错误使用

    = default 是C++11引入的特性，明确要求编译器生成默认实现：
      当你声明 = default 时，编译器会生成一个默认构造函数，不执行任何额外的初始化
      如果没有任何构造函数，编译器会自动生成一个默认构造函数
      但在这个类中已经定义了其他构造函数（看到下面有带参数的构造函数），所以需要显式请求默认构造函数
  */
  // 创建一个不执行任何额外初始化的默认构造函数
  // 明确表示该构造函数不应用于隐式类型转换
  // 类成员将使用其声明中的默认值初始化（例如 byte_size_ = 0）
  explicit Buffer() = default;

  explicit Buffer(size_t byte_size, std::shared_ptr<DeviceAllocator> allocator = nullptr,
                  void* ptr = nullptr, bool use_external = false);

  virtual ~Buffer();

  bool allocate();

  void copy_from(const Buffer& buffer) const;

  void copy_from(const Buffer* buffer) const;

  void* ptr();

  const void* ptr() const;

  size_t byte_size() const;

  std::shared_ptr<DeviceAllocator> allocator() const;

  DeviceType device_type() const;

  void set_device_type(DeviceType device_type);

  std::shared_ptr<Buffer> get_shared_from_this();

  bool is_external() const;
};
}  // namespace base

#endif