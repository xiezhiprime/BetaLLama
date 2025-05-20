#include "matmul_kernel.h"
#include "../kernels_interface.h"
#include "base/base.h"
namespace kernel {
void matmul_kernel_cpu(const tensor::Tensor& input, const tensor::Tensor& weight,
                       const tensor::Tensor& output, float scale,
                       const CudaConfig* config) {
  // UNUSED(config)表明这个CPU实现不需要CUDA配置，但函数签名保持一致是为了与GPU版本兼容。
  UNUSED(config);
  // 首先进行了一系列检查确保输入有效 
  // 这些CHECK宏可能在断言失败时会抛出异常或终止程序。
  CHECK(input.is_empty() == false);
  CHECK(weight.is_empty() == false);
  CHECK(output.is_empty() == false);
  CHECK(input.device_type() == base::DeviceType::kDeviceCPU);
  CHECK(weight.device_type() == base::DeviceType::kDeviceCPU);
  CHECK(output.device_type() == base::DeviceType::kDeviceCPU);
  // 这里获取了三个张量底层数据的指针。
  const float* input_ptr = input.ptr<float>();
  const float* weight_ptr = weight.ptr<float>();
  // 值得注意的是，虽然output_ptr被声明为常量指针，但后续会通过const_cast去除常量性以便写入数据。
  const float* output_ptr = output.ptr<float>();
  // 代码处理了输入张量可能有的两种情况
  int32_t in_dim1 = 1;
  int32_t in_dim0 = 1;
  if (input.dims_size() == 2) {
    in_dim0 = input.get_dim(0);
    in_dim1 = input.get_dim(1);
  } else if (input.dims_size() == 1) {
    in_dim0 = input.get_dim(0);
  } else {
    LOG(FATAL) << "The input tensor has a wrong dim size.";
  }
  // 检查权重是二维的：CHECK_EQ(weight.dims_size(), 2);
  CHECK_EQ(weight.dims_size(), 2);
  const int32_t wei_dim0 = weight.get_dim(0);
  const int32_t wei_dim1 = weight.get_dim(1);
  CHECK_EQ(in_dim0, wei_dim1);

  CHECK_EQ(output.size(), wei_dim0 * in_dim1);
  // 注意矩阵维度：input_mat为in_dim1 x in_dim0，weight_mat为wei_dim1 x wei_dim0

  // 创建了三个Armadillo矩阵视图（views），它们直接引用原始张量数据而不复制
  // const_cast用于去除常量性，因为Armadillo的构造函数需要非常量指针
  // Armadillo的构造函数参数是(数据指针, 行数, 列数, ...)
  arma::fmat input_mat(const_cast<float*>(input_ptr), in_dim1, in_dim0, false, true);
  arma::fmat weight_mat(const_cast<float*>(weight_ptr), wei_dim1, wei_dim0, false, true);
  arma::fmat output_mat(const_cast<float*>(output_ptr), in_dim1, wei_dim0, false, true);
  // 这种维度转置处理可能是为了适应不同的内存布局（如列优先存储）或计算库之间的接口差异。
  // 在这种情况下，矩阵乘法要求input_mat的列数(in_dim0)必须等于weight_mat的行数(wei_dim1)，
  // 所以检查CHECK_EQ(in_dim0, wei_dim1)是正确的。
  output_mat = ((input_mat * weight_mat)) * scale;
}
}  // namespace kernel