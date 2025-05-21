#include "swiglu_kernel.h"
namespace kernel {
void swiglu_kernel_cpu(const tensor::Tensor& input1, const tensor::Tensor& input2,
                       const tensor::Tensor& output, void* stream) {
  // 通常用于 GPU 计算中的流控制，在 CPU 实现中未使用 (UNUSED(stream))
  UNUSED(stream);
  CHECK_EQ(input1.is_empty(),false);
  CHECK_EQ(input2.is_empty(),false);
  CHECK_EQ(output.is_empty(),false);

  CHECK(input1.device_type() == base::DeviceType::kDeviceCPU);
  CHECK(input2.device_type() == base::DeviceType::kDeviceCPU);
  CHECK(output.device_type() == base::DeviceType::kDeviceCPU);
  
  // 代码创建了 Armadillo 向量视图来访问张量数据
  // 第一个参数：指向数据的指针 (使用 const_cast 移除常量性)
  // 第二个参数：向量大小
  // 第三个参数 false：指示不复制数据
  // 第四个参数 true：指示这是对现有内存的引用，不应在 fvec 析构时释放内存
  arma::fvec input1_vec(const_cast<float*>(input1.ptr<float>()), input1.size(), false,
                        true);
  arma::fvec input2_vec(const_cast<float*>(input2.ptr<float>()), input2.size(), false,
                        true);
  arma::fvec output_vec(const_cast<float*>(output.ptr<float>()), output.size(), false,
                        true);
  // SwiGLU(x, y) = Swish(x) ⊙ y
  
  // 这里的 %= 是 Armadillo 中的逐元素相乘赋值运算符（类似于 *=，但是是逐元素操作）
  input1_vec %= (1.0f / (1.0f + arma::exp(-input1_vec)));
  // 这里的 % 是 Armadillo 中的逐元素乘法运算符（Hadamard 积）
  output_vec = input1_vec % input2_vec;
}
}  // namespace kernel