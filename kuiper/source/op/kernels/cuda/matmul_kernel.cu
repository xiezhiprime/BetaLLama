#include <tensor/tensor.h>
#include <cub/block/block_reduce.cuh>
#include "../kernels_interface.h"
#include "matmul_kernel.cuh"
namespace kernel {
// THREAD_PER_BLOCK（每个块的线程数）和ROW_PER_BLOCK（每个块处理的行数）
template <int THREAD_PER_BLOCK, int ROW_PER_BLOCK>
__global__ void matmul_kernel_cu_fp32(const float* input, const float* weight, float* output, int M,
                                      int K) {
  /*
      这个核函数实现的是矩阵-向量乘法：output = weight × input
      
      weight（权重）：形状为(K, M) - 一个二维矩阵，有K行M列
      input（输入）：形状为(M) - 实际上是一个长度为M的一维向量
      output（输出）：形状为(K) - 一个长度为K的一维向量
  */
  // 声明块内共享内存数组，用于线程间通信和临时结果存储                                      
  __shared__ float sdata[THREAD_PER_BLOCK];
  // 获取当前线程在块内的索引，用于识别线程职责
  unsigned int tid = threadIdx.x;

  // 基于块ID计算该块负责的输出矩阵行范围, 每个块处理从start_row到end_row-1的多行
  int start_row = blockIdx.x * ROW_PER_BLOCK;
  int end_row = start_row + ROW_PER_BLOCK;
  if (start_row >= K) {
    // 超出范围的块直接返回，避免越界计算
    return;
  }

  // 每次处理4个浮点数（对应float4数据类型）
  constexpr int pack_size = 4;
  // 可以完整打包的组数（M除以4的商）
  const int pack_num = M / pack_size;
  // 所有完整打包后的偏移量，用于后续处理剩余元素
  const int pack_off = pack_size * pack_num;

// 指示编译器展开循环，减少循环开销
#pragma unroll
  // 每个块依次处理分配给它的每一行输出
  for (int p = start_row; p < end_row; ++p) {
    // 初始化线程的累加变量为0
    sdata[tid] = 0;
    // 计算当前行在权重矩阵中的偏移量
    int row_offset = p * M;
    // 将输入和权重数据重新解释为float4类型指针，以便进行向量化读取
    float4* input_float4_ptr = (float4*)input;
    float4* weight_float4_ptr = (float4*)(weight + row_offset);

    // 这是核心计算部分, 这种向量化方法大幅减少了内存访问次数
#pragma unroll
    // 每个线程跳跃式（stride）处理不同的数据块
    for (int i = tid; i < pack_num; i += blockDim.x) {
      // 使用float4一次读取4个浮点数，显著提高内存带宽效率
      float4 input_float4 = *(input_float4_ptr + i);
      float4 weight_float4 = *(weight_float4_ptr + i);
      float part_sum = input_float4.x * weight_float4.x + input_float4.y * weight_float4.y +
                       input_float4.z * weight_float4.z + input_float4.w * weight_float4.w;
      // 计算4对元素的点积，并累加到线程的局部和中
      sdata[tid] += part_sum;
    }

    // 处理无法整除4的剩余元素
    for (int i = pack_off + tid; i < M; i += blockDim.x) {
      // 从pack_off（已处理的向量元素后）开始, 使用标准的逐元素乘法
      sdata[tid] += input[i] * weight[row_offset + i];
    }
    // 确保所有线程完成计算后再进行归约
    __syncthreads();
    // 使用NVIDIA CUB库的BlockReduce高效实现块内并行求和, 这种归约算法比手动实现更高效，利用了GPU架构特性
    using BlockReduce = cub::BlockReduce<float, THREAD_PER_BLOCK>;
    __shared__ typename BlockReduce::TempStorage temp;
    float part_sum = BlockReduce(temp).Sum(sdata[tid]);
    __syncthreads();
    // 只有线程0负责将最终结果写入全局内存
    if (tid == 0) {
      output[p] = part_sum;
    }
    // __syncthreads()确保当前行处理完毕后再处理下一行
    __syncthreads();
  }
}

template <int THREAD_PER_BLOCK, int ROW_PER_BLOCK>
__global__ void matmul_kernel_cu_fp32int8(const float* input, const int8_t* weight,
                                          const float* scales, const int32_t group_size,
                                          float* output, int M, int K) {
  __shared__ float sdata[THREAD_PER_BLOCK];
  unsigned int tid = threadIdx.x;

  int start_row = blockIdx.x * ROW_PER_BLOCK;
  int end_row = start_row + ROW_PER_BLOCK;
  if (start_row >= K) {
    return;
  }
  for (int p = start_row; p < end_row; ++p) {
    sdata[tid] = 0;
    for (int i = tid; i < M; i += THREAD_PER_BLOCK) {
      const int weight_idx = p * M + i;
      const int group_idx = weight_idx / group_size;
      sdata[tid] += input[i] * scales[group_idx] * static_cast<float>(weight[weight_idx]);
    }
    __syncthreads();

    using BlockReduce = cub::BlockReduce<float, THREAD_PER_BLOCK>;
    __shared__ typename BlockReduce::TempStorage temp;
    float part_sum = BlockReduce(temp).Sum(sdata[tid]);
    __syncthreads();

    if (tid == 0) {
      output[p] = part_sum;
    }
    __syncthreads();
  }
}

void matmul_kernel_cu(const tensor::Tensor& input, const tensor::Tensor& weight,
                      const tensor::Tensor& output, const float scale, const CudaConfig* config) {
  CHECK(input.is_empty() == false && input.dims_size() <= 2);
  CHECK(input.device_type() == base::DeviceType::kDeviceCUDA);

  CHECK(weight.is_empty() == false && weight.dims_size() == 2);
  CHECK(weight.device_type() == base::DeviceType::kDeviceCUDA);
  // weight : (K, M)
  const int32_t K = weight.get_dim(0);  // row
  const int32_t M = weight.get_dim(1);  // col
  int packet_size = 4;
  // CHECK_EQ(M % packet_size, 0);
  // input : (M)
  CHECK_EQ(M, input.get_dim(0));
  if (config && config->stream) {
    matmul_kernel_cu_fp32<128, 1><<<K, 128, 0, config->stream>>>(
        input.ptr<float>(), weight.ptr<float>(), const_cast<float*>(output.ptr<float>()), M, K);
  } else {
    matmul_kernel_cu_fp32<128, 1><<<K, 128>>>(input.ptr<float>(), weight.ptr<float>(),
                                              const_cast<float*>(output.ptr<float>()), M, K);
  }
}

void matmul_kernel_cu_qint8(const tensor::Tensor& input, const tensor::Tensor& weight,
                            const tensor::Tensor& output, int32_t group_size,
                            const tensor::Tensor& scale, const CudaConfig* config) {
  CHECK(config != nullptr);
  CHECK(input.is_empty() == false && input.dims_size() <= 2);
  CHECK(input.device_type() == base::DeviceType::kDeviceCUDA);

  CHECK(weight.is_empty() == false && weight.dims_size() == 2);
  CHECK(weight.device_type() == base::DeviceType::kDeviceCUDA);
  const int32_t K = weight.get_dim(0);  // row
  const int32_t M = weight.get_dim(1);  // col
  int packet_size = 4;
  CHECK_EQ(M % packet_size, 0);
  CHECK_EQ(M, input.get_dim(0));
  if (config->stream) {
    matmul_kernel_cu_fp32int8<128, 1><<<K, 128, 0, config->stream>>>(
        input.ptr<float>(), weight.ptr<int8_t>(), scale.ptr<float>(), group_size,
        const_cast<float*>(output.ptr<float>()), M, K);
  } else {
    matmul_kernel_cu_fp32int8<128, 1><<<K, 128>>>(input.ptr<float>(), weight.ptr<int8_t>(),
                                                  scale.ptr<float>(), group_size,
                                                  const_cast<float*>(output.ptr<float>()), M, K);
  }
}
}  // namespace kernel