#include <glog/logging.h>
#include <cstdlib>
#include "base/alloc.h"

#if (defined(_POSIX_ADVISORY_INFO) && (_POSIX_ADVISORY_INFO >= 200112L))
#define KUIPER_HAVE_POSIX_MEMALIGN
#endif

namespace base {
CPUDeviceAllocator::CPUDeviceAllocator() : DeviceAllocator(DeviceType::kDeviceCPU) {
}

void* CPUDeviceAllocator::allocate(size_t byte_size) const {
  // 这里首先检查请求分配的字节数是否为0。如果是0，直接返回空指针。这是一个很好的防御性编程实践，避免无意义的内存分配操作。
  if (!byte_size) {
    return nullptr;
  }
// 根据编译时的环境配置，会选择两种不同的内存分配策略。
#ifdef KUIPER_HAVE_POSIX_MEMALIGN
  void* data = nullptr;
  // 这里有个很巧妙的设计：动态对齐策略。代码根据分配大小来决定内存对齐的字节数：
  // 如果要分配的内存≥1024字节，使用32字节对齐 ： 小块内存使用16字节对齐就足够了，可以充分利用现代CPU的缓存行特性
  // 如果要分配的内存<1024字节，使用16字节对齐 ： 大块内存使用32字节对齐能更好地配合向量化指令（如AVX指令集）
  const size_t alignment = (byte_size >= size_t(1024)) ? size_t(32) : size_t(16);
  // 调用posix_memalign函数进行对齐内存分配。 保护措施：确保对齐值至少等于指针的大小（sizeof(void*)），这是POSIX标准的要求。
  int status = posix_memalign((void**)&data,
                              ((alignment >= sizeof(void*)) ? alignment : sizeof(void*)),
                              byte_size);
  if (status != 0) {
    return nullptr;
  }
  return data;
#else
  // 这是最基本的内存分配方式，虽然不能保证内存对齐，但具有最广泛的兼容性。
  void* data = malloc(byte_size);
  return data;
#endif
}

void CPUDeviceAllocator::release(void* ptr) const {
  if (ptr) {
    free(ptr);
  }
}

std::shared_ptr<CPUDeviceAllocator> CPUDeviceAllocatorFactory::instance = nullptr;
}  // namespace base