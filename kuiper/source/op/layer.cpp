#include "op/layer.h"
#include <base/cuda_config.h>
#include <glog/logging.h>
#include <cstdarg>
#include <numeric>
#include <utility>

namespace op {
BaseLayer::BaseLayer(base::DeviceType device_type, LayerType layer_type, base::DataType data_type,
                     std::string layer_name)
    : device_type_(device_type),
      layer_type_(layer_type),
      data_type_(data_type),
      layer_name_(std::move(layer_name)) {}

base::DataType BaseLayer::data_type() const { return data_type_; }

LayerType BaseLayer::layer_type() const { return layer_type_; }

base::Status BaseLayer::set_weight(int32_t idx, const tensor::Tensor& weight) {
  return base::error::FunctionNotImplement();
}

base::Status BaseLayer::set_weight(int32_t idx, const std::vector<int32_t>& dims,
                                   const void* weight_ptr, base::DeviceType device_type) {
  return base::error::FunctionNotImplement();
}

const std::string& BaseLayer::get_layer_name() const { return layer_name_; }

void BaseLayer::set_layer_name(const std::string& layer_name) { layer_name_ = layer_name; }
base::DeviceType BaseLayer::device_type() const { return device_type_; }

void BaseLayer::set_device_type(base::DeviceType device_type) { device_type_ = device_type; }

Layer::Layer(base::DeviceType device_type, LayerType layer_type, std::string layer_name)
    : BaseLayer(device_type, layer_type, base::DataType::kDataTypeFp32, std::move(layer_name)) {}

base::Status Layer::init() { return base::error::Success(); }

base::Status Layer::forward() { return base::error::FunctionNotImplement(""); }

base::Status Layer::check_tensor(const tensor::Tensor& tensor, base::DeviceType device_type,
                                 base::DataType data_type) const {
  if (tensor.is_empty()) {
    return base::error::InvalidArgument("The tensor parameter is empty.");
  }
  if (tensor.device_type() != device_type) {
    return base::error::InvalidArgument("The tensor has a wrong device type.");
  }
  if (tensor.data_type() != data_type) {
    return base::error::InvalidArgument("The tensor has a wrong data type.");
  }
  return base::error::Success();
}

base::Status Layer::check_tensor_with_dim(const tensor::Tensor& tensor,
                                          base::DeviceType device_type, base::DataType data_type,
                                          ...) const {
  std::va_list args;
  // 这个函数允许灵活地验证张量维度
  // 这种设计在机器学习框架中特别有用，
  // 能够简洁地验证输入张量是否符合模型期望的形状，避免后续计算出现错误。
  if (tensor.is_empty()) {
    return base::error::InvalidArgument("The tensor parameter is empty.");
  }
  if (tensor.device_type() != device_type) {
    return base::error::InvalidArgument("The tensor has a wrong device type.");
  }
  if (tensor.data_type() != data_type) {
    return base::error::InvalidArgument("The tensor has a wrong data type.");
  }

  va_start(args, data_type);
  int32_t dims = tensor.dims_size();
  // 循环遍历张量的每个维度
  for (int32_t i = 0; i < dims; ++i) {
    // 提取下一个参数值（预期为int32_t类型）
    int32_t dim = va_arg(args, int32_t);
    if (dim != tensor.get_dim(i)) {
      return base::error::InvalidArgument("The tensor has a wrong dim in dim" + std::to_string(i));
    }
  }
  // 完成可变参数处理，释放资源
  va_end(args);
  return base::error::Success();
}

void Layer::set_input(int32_t idx, const tensor::Tensor& input) {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, inputs_.size());
  this->inputs_.at(idx) = input;
}

void Layer::set_output(int32_t idx, const tensor::Tensor& output) {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, outputs_.size());
  this->outputs_.at(idx) = output;
}

const tensor::Tensor& Layer::get_input(int32_t idx) const {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, inputs_.size());
  return inputs_.at(idx);
}

tensor::Tensor& Layer::get_input(int32_t idx) {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, inputs_.size());
  return inputs_.at(idx);
}

tensor::Tensor& Layer::get_output(int32_t idx) {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, outputs_.size());
  return outputs_.at(idx);
}

base::Status Layer::check() const {
  return base::error::FunctionNotImplement("The check function is not implement yet");
}

const tensor::Tensor& Layer::get_output(int32_t idx) const {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, outputs_.size());
  return outputs_.at(idx);
}

void Layer::reset_input_size(size_t size) { inputs_.resize(size); }

void Layer::reset_output_size(size_t size) { outputs_.resize(size); }

void Layer::to_cuda() {
  for (auto& input : inputs_) {
    if (!input.is_empty()) {
      input.to_cuda(cuda_config_ ? cuda_config_->stream : nullptr);
    }
  }
  for (auto& output : outputs_) {
    if (!output.is_empty()) {
      output.to_cuda(cuda_config_ ? cuda_config_->stream : nullptr);
    }
  }
}

void Layer::set_cuda_config(std::shared_ptr<kernel::CudaConfig> config) {
  if (!config) {
    return;
  }
  this->cuda_config_ = config;
}

std::shared_ptr<kernel::CudaConfig> Layer::cuda_config() const { return cuda_config_; }

size_t Layer::input_size() const { return inputs_.size(); }

size_t Layer::output_size() const { return outputs_.size(); }

LayerParam::LayerParam(base::DeviceType device_type, LayerType layer_type, bool is_quant_layer,
                       std::string layer_name)
    : Layer(device_type, layer_type, std::move(layer_name)), is_quant_layer_(is_quant_layer) {}

base::Status LayerParam::set_weight(int32_t idx, const tensor::Tensor& weight) {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, weights_.size());
  CHECK(weight.data_type() == base::DataType::kDataTypeFp32);
  if (!weight.is_empty()) {
    CHECK(weight.device_type() == device_type_);
  }
  weights_.at(idx) = weight;
  return base::error::Success();
}

const tensor::Tensor& LayerParam::get_weight(int32_t idx) const {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, weights_.size());
  return weights_.at(idx);
}

void LayerParam::to_cuda() {
  Layer::to_cuda();
  for (auto& weight : weights_) {
    weight.to_cuda(cuda_config_ ? cuda_config_->stream : nullptr);
  }
  if (!scales_.is_empty()) {
    scales_.to_cuda(cuda_config_ ? cuda_config_->stream : nullptr);
  }
}

base::Status LayerParam::set_weight(int32_t idx, const std::vector<int32_t>& dims,
                                    const void* weight_ptr, base::DeviceType device_type) {
  
  /*
    idx：权重索引，指定要设置的是哪个权重（一个层可能有多个权重）
    dims：权重的维度信息，如[词汇表大小, 嵌入维度]
    weight_ptr：指向权重数据的原始指针
    device_type：指定设备类型（CPU/GPU等）
  */
  // 检查：索引非负，索引在有效范围内，权重指针不为空
  CHECK_GE(idx, 0);
  CHECK_LT(idx, weights_.size());
  CHECK_NE(weight_ptr, nullptr);

  // 2. 计算内存大小并创建Buffer
  // 计算内存大小：使用std::accumulate和std::multiplies计算所有维度的乘积，再乘以sizeof(float)得到总字节数
  size_t size = std::accumulate(dims.begin(), dims.end(), sizeof(float), std::multiplies<>());
  // 创建零拷贝Buffer：注意最后一个参数true，这表明创建的是一个"零拷贝"Buffer，
  // 它不会复制weight_ptr指向的数据，而是直接引用它.
  std::shared_ptr<base::Buffer> buffer =
      std::make_shared<base::Buffer>(size, nullptr, const_cast<void*>(weight_ptr), true);
  if (device_type != base::DeviceType::kDeviceUnknown) {
    buffer->set_device_type(device_type);
  }

  if (!is_quant_layer_) {
    // 非量化层处理（浮点权重）
    // 创建一个32位浮点类型的张量，形状为指定的dims
    tensor::Tensor weight(base::DataType::kDataTypeFp32, dims);
    weight.set_device_type(device_type);
    // 将刚才创建的buffer关联到张量
    CHECK(weight.assign(buffer));
    // 将张量存储到weights_容器的指定位置
    weights_.at(idx) = weight;
    /*
      在CPU场景下，整个链路从mmap到最终的Tensor是完全零拷贝的，
      所有对象都直接或间接地引用最初映射的内存，形成了这样的引用链：
          映射内存空间 <- weight_ptr <- Buffer <- Tensor <- weights_容器中的元素

      当设备类型为GPU时，情况就会有所不同：
        这时候会发生什么呢？由于权重数据需要在GPU上使用，
        但原始数据位于CPU内存（通过mmap映射的内存空间），所以必然需要进行一次CPU到GPU的数据传输。
        这个传输通常在weight.assign(buffer)中实现
        此时的引用链变成了：
          映射内存空间 -> 复制 -> GPU内存 <- Tensor <- weights_容器中的元素
    */
  } else {
    // is quant layer
    tensor::Tensor weight(base::DataType::kDataTypeInt8, dims);
    weight.set_device_type(device_type);
    CHECK(weight.assign(buffer));
    weights_.at(idx) = weight;

    const int32_t weight_size = static_cast<int32_t>(weight.size());
    CHECK(weight_size % group_size_ == 0);

    int32_t scale_nums = weight_size / group_size_;
    scales_ = tensor::Tensor{base::DataType::kDataTypeFp32, scale_nums, false, nullptr,
                             reinterpret_cast<float*>((int8_t*)weight_ptr + weight_size)};
    scales_.set_device_type(device_type);
  }

  return base::error::Success();
}

void LayerParam::set_scales(const tensor::Tensor& scales) {
  CHECK(!scales.is_empty());
  this->scales_ = scales;
}

void LayerParam::set_group_size(int32_t group_size) { this->group_size_ = group_size; }

int32_t LayerParam::get_scale_num() const {
  CHECK(!scales_.is_empty());
  return static_cast<int32_t>(scales_.size());
}

void LayerParam::reset_weight_size(size_t size) { weights_.resize(size); }

size_t LayerParam::weight_size() const { return weights_.size(); }

base::Status Layer::forward(const tensor::Tensor& input1, const tensor::Tensor& output1) {
  this->set_input(0, input1);
  this->set_output(0, output1);
  return this->forward();
}

base::Status Layer::forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                            const tensor::Tensor& output1) {
  this->set_input(0, input1);
  this->set_input(1, input2);

  this->set_output(0, output1);
  return this->forward();
}

base::Status Layer::forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                            const tensor::Tensor& input3, const tensor::Tensor& output1) {
  this->set_input(0, input1);
  this->set_input(1, input2);
  this->set_input(2, input3);

  this->set_output(0, output1);
  return this->forward();
}

base::Status Layer::forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                            const tensor::Tensor& input3, const tensor::Tensor& input4,
                            const tensor::Tensor& output1) {
  this->set_input(0, input1);
  this->set_input(1, input2);
  this->set_input(2, input3);
  this->set_input(3, input4);

  this->set_output(0, output1);
  return this->forward();
}

base::Status Layer::forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                            const tensor::Tensor& input3, const tensor::Tensor& input4,
                            const tensor::Tensor& input5, const tensor::Tensor& output1) {
  this->set_input(0, input1);
  this->set_input(1, input2);
  this->set_input(2, input3);
  this->set_input(3, input4);
  this->set_input(4, input5);

  this->set_output(0, output1);
  return this->forward();
}

tensor::Tensor& LayerParam::get_weight(int32_t idx) {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, weights_.size());
  return weights_.at(idx);
}

}  // namespace op