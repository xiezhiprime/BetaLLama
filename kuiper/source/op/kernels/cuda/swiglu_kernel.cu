#include <tensor/tensor.h>
#include "swiglu_kernel.cuh"
namespace kernel {
__global__ void swiglu_kernel_cu_fp32(int size, const float* in1, const float* in2, float* out) {
  // 标记局部线程id
  int tid = threadIdx.x;
  // 标记全局线程id 线程在整个网格中的全局ID，用于访问对应的数组元素
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  // 这行确保不会处理超出数组大小的元素，这是CUDA编程中的常见做法，因为通常会启动"多余"的线程以对齐线程块大小。
  if (idx >= size) {
    return;
  }
  extern __shared__ float shared_mem[];
  // 第一个区域从shared_mem开始，大小为blockDim.x * sizeof(float)字节
  float* smem1 = shared_mem;
  // 第二个区域从shared_mem + blockDim.x开始，同样大小为blockDim.x * sizeof(float)字节
  float* smem2 = shared_mem + blockDim.x;
  // 这种方法允许我们在一个共享内存块中存储两个数组， 
  // 这比声明两个独立的共享内存数组更高效，因为它避免了可能的内存对齐问题和额外的管理开销。

  // 将数据从全局内存搬到共享内存： 共享内存比全局内存快得多(通常快100倍)，因为它位于芯片上并且被同一块内的所有线程共享。通过将频繁访问的数据从全局内存加载到共享内存，我们可以显著减少内存访问延迟，提高计算效率。
  smem1[tid] = in1[idx];
  smem2[tid] = in2[idx];
  // 进行同步
  __syncthreads();
  // 每个线程操作自己在共享内存中的数据
  float value = 1.0f / (1.0f + exp(-smem1[tid]));  // sigmoid
  smem1[tid] = smem1[tid] * value;  // Swish
  // 将共享内存中的数据搬回全局内存
  out[idx] = smem1[tid] * smem2[tid];  // SwiGLU
}

void swiglu_kernel_cu(const tensor::Tensor& input1, const tensor::Tensor& input2,
                      const tensor::Tensor& output, void* stream) {
  // 代码首先验证所有张量不为空且都在CUDA设备上
  CHECK_EQ(input1.is_empty(), false);
  CHECK(input1.device_type() == base::DeviceType::kDeviceCUDA);

  CHECK_EQ(input2.is_empty(), false);
  CHECK(input2.device_type() == base::DeviceType::kDeviceCUDA);

  CHECK_EQ(output.is_empty(), false);
  CHECK(output.device_type() == base::DeviceType::kDeviceCUDA);
  
  // 定义了CUDA并行执行的配置
  int size = static_cast<int32_t>(input1.size()); // 要处理的元素总数
  int threads = 128;
  int blocks = (size + threads - 1) / threads; // 需要的线程块数量，使用向上取整的除法确保能处理所有元素
  const size_t shmem = threads * sizeof(float) * 2; // 分配的共享内存大小（每个线程2个float值）
  if (!stream) {
    swiglu_kernel_cu_fp32<<<blocks, threads, shmem>>>(
        size, input1.ptr<float>(), input2.ptr<float>(), const_cast<float*>(output.ptr<float>()));
  } else {
    cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
    swiglu_kernel_cu_fp32<<<blocks, threads, shmem, stream_>>>(
        size, input1.ptr<float>(), input2.ptr<float>(), const_cast<float*>(output.ptr<float>()));
  }
}
}  // namespace kernel