#include "model/llama3.h"
#include <cuda_runtime_api.h>
#include <glog/logging.h>
#include <op/matmul.h>
#include <op/mha.h>
#include <op/rmsnorm.h>
#include <sentencepiece_processor.h>
#include <utility>
#include "../op/kernels/cpu/rope_kernel.h"
#include "../op/kernels/cuda/rope_kernel.cuh"
#include "base/tick.h"
namespace model {

void LLama2Layers::to_cuda(std::shared_ptr<kernel::CudaConfig> config) {
  if (add_layer_) {
    add_layer_->set_cuda_config(config);
    add_layer_->to_cuda();
  }

  if (rope_layer_) {
    rope_layer_->set_cuda_config(config);
    rope_layer_->to_cuda();
  }

  if (swiglu_layer_) {
    swiglu_layer_->set_cuda_config(config);
    swiglu_layer_->to_cuda();
  }

  if (cls_layer_) {
    cls_layer_->set_cuda_config(config);
    cls_layer_->to_cuda();
  }

  if (embedding_layer_) {
    embedding_layer_->set_cuda_config(config);
    embedding_layer_->to_cuda();
  }

  if (mha_layer_) {
    mha_layer_->set_cuda_config(config);
    mha_layer_->to_cuda();
  }

  for (auto& weight_layer : wq_layers_) {
    if (weight_layer) {
      weight_layer->set_cuda_config(config);
      weight_layer->to_cuda();
    }
  }

  for (auto& weight_layer : wk_layers_) {
    if (weight_layer) {
      weight_layer->set_cuda_config(config);
      weight_layer->to_cuda();
    }
  }

  for (auto& weight_layer : wv_layers_) {
    if (weight_layer) {
      weight_layer->set_cuda_config(config);
      weight_layer->to_cuda();
    }
  }

  for (auto& weight_layer : wo_layers_) {
    if (weight_layer) {
      weight_layer->set_cuda_config(config);
      weight_layer->to_cuda();
    }
  }

  for (auto& weight_layer : w1_layers_) {
    if (weight_layer) {
      weight_layer->set_cuda_config(config);
      weight_layer->to_cuda();
    }
  }

  for (auto& weight_layer : w2_layers_) {
    if (weight_layer) {
      weight_layer->set_cuda_config(config);
      weight_layer->to_cuda();
    }
  }

  for (auto& weight_layer : w3_layers_) {
    if (weight_layer) {
      weight_layer->set_cuda_config(config);
      weight_layer->to_cuda();
    }
  }

  for (auto& rms_norm_layer : rmsnorm_layers_) {
    if (rms_norm_layer) {
      rms_norm_layer->to_cuda();
      rms_norm_layer->set_cuda_config(config);
    }
  }
}

LLama2Model::LLama2Model(base::TokenizerType tokenizer_type, std::string token_path,
                         std::string model_path, bool is_quant_model)
    : Model(tokenizer_type, base::ModelType::kModelTypeLLama2, std::move(token_path),
            std::move(model_path), is_quant_model) {}

/*
  一个完整的模型初始化流程，涵盖了从设备选择、资源分配到模型参数加载的所有环节。
  特别值得注意的是，代码对CPU和GPU两种运行环境提供了不同的实现路径，并进行了必要的错误检查，确保模型能够在正确的环境中启动。
*/
base::Status LLama2Model::init(base::DeviceType device_type) {
  using namespace base;
  // 确保分词器路径不为空，这是模型处理文本的必要组件
  if (token_path_.empty()) {
    return error::PathNotValid(token_path_);
  }
  // 验证如果使用CPU时不能运行量化（int8）模型，这是一个硬件兼容性限制
  if (device_type == base::DeviceType::kDeviceCPU && is_quant_model_) {
    return error::InternalError("The cpu device do not support int8 quant model.");
  }

  device_type_ = device_type;
  if (device_type == DeviceType::kDeviceCUDA) {
    // 选择系统中的第一个CUDA设备（GPU 0）
    cudaSetDevice(0);
    // 创建CUDA配置对象
    cuda_config_ = std::make_shared<kernel::CudaConfig>();
    // 创建CUDA流以支持异步操作，这对于高效处理大型矩阵计算非常重要
    cudaStreamCreate(&cuda_config_->stream);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      return error::InternalError("The cuda hanle create failed.");
    }
  }
  
  // 从checkpoint文件加载预训练模型参数， 这个函数中创建了网络的各个层，以及使用mmap将参数从文件中映射到程序中。
  Status read_status = gen_model_from_file();
  if (!read_status) {
    return read_status;
  }

  // 负责为模型分配和初始化必要的内存缓冲区，为后续的模型计算做准备。
  init_mem();
  // 这段代码计算正弦和余弦缓存，用于transformer模型的位置编码 : 位置编码允许模型理解序列中token的相对位置，这对于语言理解至关重要
  if (device_type_ == base::DeviceType::kDeviceCPU) {
    kernel::sin_cos_cache_calc_cpu(config_->head_size_, config_->seq_len_,
                                   get_buffer(ModelBufferType::kSinCache).ptr<float>(),
                                   get_buffer(ModelBufferType::kCosCache).ptr<float>());
  } else {
    CHECK_NE(cuda_config_, nullptr);
    kernel::sin_cos_cache_calc_cu(config_->head_size_, config_->seq_len_,
                                  get_buffer(ModelBufferType::kSinCache),
                                  get_buffer(ModelBufferType::kCosCache), cuda_config_->stream);
  }
  // 创建一个ArgmaxSampler（最大值采样器）实例，用于文本生成时选择最可能的下一个词元
  sampler_ = std::make_unique<sampler::ArgmaxSampler>(device_type_);
  return error::Success();
}

base::Status LLama2Model::forward(const tensor::Tensor& input, const tensor::Tensor& pos_tensor,
                                  int& next) const {
  if (input.is_empty()) {
    return base::error::InvalidArgument("The input tensor is empty.");
  }
  if (device_type_ == base::DeviceType::kDeviceCPU && is_quant_model_) {
    return base::error::InternalError("Unsupported int8 quant in the cpu device");
  }

  for (int32_t layer_idx = 0; layer_idx < config_->layer_num_; ++layer_idx) {
    attention_rms(layer_idx, input);
    // attention (wq wk wv @ input)
    attention_qkv(layer_idx, pos_tensor);
    // multi-head attention
    attention_mha(layer_idx, pos_tensor);
    // feed forward
    feed_forward(layer_idx, input);
  }
  cls_logits(input);
  return base::error::Success();
}

void LLama2Model::create_nonparam_layers() {
  CHECK(llama_layers_ != nullptr);
  llama_layers_->rope_layer_ = std::make_shared<op::RoPELayer>(
      device_type_, config_->dim_, config_->kv_dim_, config_->head_size_);

  llama_layers_->mha_layer_ = std::make_shared<op::MultiHeadAttention>(
      device_type_, 0, config_->kv_mul_, config_->kv_dim_, config_->seq_len_, config_->head_num_,
      config_->head_size_);

  llama_layers_->add_layer_ = std::make_shared<op::VecAddLayer>(device_type_);

  llama_layers_->swiglu_layer_ =
      std::make_shared<op::SwiGLULayer>(device_type_, config_->hidden_dim_);
}

void LLama2Model::create_param_quant_layers() {
  CHECK(is_quant_model_);
  CHECK(llama_layers_ != nullptr);

  size_t pos = 0;
  int32_t dim = config_->dim_;
  auto cpu_device_type = base::DeviceType::kDeviceCPU;

  // query
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto wq = std::make_shared<op::MatmulLayer>(device_type_, dim, dim, true);
    wq->set_group_size(group_size_);
    wq->set_weight(0, {dim, dim}, this->raw_model_data_->weight(pos), cpu_device_type);
    llama_layers_->wq_layers_.push_back(wq);
    pos = pos + dim * dim + wq->get_scale_num() * sizeof(float);
  }

  // key
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto wk = std::make_shared<op::MatmulLayer>(device_type_, config_->kv_dim_, dim, true);
    wk->set_group_size(group_size_);
    wk->set_weight(0, {config_->kv_dim_, dim}, this->raw_model_data_->weight(pos), cpu_device_type);
    llama_layers_->wk_layers_.push_back(wk);
    pos = pos + config_->kv_dim_ * dim + wk->get_scale_num() * sizeof(float);
  }

  // value
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto wv = std::make_shared<op::MatmulLayer>(device_type_, config_->kv_dim_, dim, true);
    wv->set_group_size(group_size_);
    wv->set_weight(0, {config_->kv_dim_, dim}, this->raw_model_data_->weight(pos), cpu_device_type);
    llama_layers_->wv_layers_.push_back(wv);
    pos += config_->kv_dim_ * dim + wv->get_scale_num() * sizeof(float);
  }

  // output
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto wo = std::make_shared<op::MatmulLayer>(device_type_, dim, dim, true);
    wo->set_group_size(group_size_);
    wo->set_weight(0, {dim, dim}, this->raw_model_data_->weight(pos), cpu_device_type);
    llama_layers_->wo_layers_.push_back(wo);
    pos = pos + dim * dim + wo->get_scale_num() * sizeof(float);
  }

  // w1 layers
  int32_t hidden_dim = config_->hidden_dim_;
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto w1 = std::make_shared<op::MatmulLayer>(device_type_, hidden_dim, dim, true);
    w1->set_group_size(group_size_);
    w1->set_weight(0, {hidden_dim, dim}, this->raw_model_data_->weight(pos), cpu_device_type);
    llama_layers_->w1_layers_.push_back(w1);
    pos = pos + dim * hidden_dim + w1->get_scale_num() * sizeof(float);
  }

  // w2 layers
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto w2 = std::make_shared<op::MatmulLayer>(device_type_, dim, hidden_dim, true);
    w2->set_group_size(group_size_);
    w2->set_weight(0, {dim, hidden_dim}, this->raw_model_data_->weight(pos), cpu_device_type);
    llama_layers_->w2_layers_.push_back(w2);
    pos = pos + dim * hidden_dim + w2->get_scale_num() * sizeof(float);
  }

  // w3 layers
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto w3 = std::make_shared<op::MatmulLayer>(device_type_, hidden_dim, dim, true);
    w3->set_group_size(group_size_);
    w3->set_weight(0, {hidden_dim, dim}, this->raw_model_data_->weight(pos), cpu_device_type);
    llama_layers_->w3_layers_.push_back(w3);
    pos = pos + dim * hidden_dim + w3->get_scale_num() * sizeof(float);
  }

  // wcls layer
  auto cls_layer = std::make_shared<op::MatmulLayer>(device_type_, config_->vocab_size_, dim, true);
  cls_layer->set_group_size(group_size_);
  if (config_->is_shared_weight_) {
    // using token embedding weight
    cls_layer->set_weight(0, {config_->vocab_size_, dim}, this->raw_model_data_->weight(pos),
                          cpu_device_type);
  } else {
    // no shared
    cls_layer->set_weight(0, {config_->vocab_size_, dim}, this->raw_model_data_->weight(pos),
                          cpu_device_type);
    pos = pos + config_->vocab_size_ * dim + cls_layer->get_scale_num() * sizeof(float);
  }
  llama_layers_->cls_layer_ = cls_layer;

  // embedding layer
  float* weight_ptr = (float*)raw_model_data_->weight(pos);
  llama_layers_->embedding_layer_ = std::make_shared<op::EmbeddingLayer>(
      device_type_, config_->dim_, config_->seq_len_, std::abs(config_->vocab_size_));
  llama_layers_->embedding_layer_->set_weight(0, {std::abs(config_->vocab_size_), dim}, weight_ptr,
                                              cpu_device_type);
  weight_ptr += config_->vocab_size_ * dim;

  // rmsnorm attention attention,ffn,final
  for (int32_t i = 0; i < 2 * config_->layer_num_ + 1; ++i) {
    std::shared_ptr<op::RmsNormLayer> rms_norm_layer =
        std::make_shared<op::RmsNormLayer>(device_type_, dim);

    rms_norm_layer->set_weight(0, {dim}, weight_ptr, cpu_device_type);
    llama_layers_->rmsnorm_layers_.push_back(rms_norm_layer);
    weight_ptr += dim;
  }
}

// 这个函数实现了Llama2模型参数层的创建过程，负责从预加载的权重数据中构建完整的神经网络架构。
void LLama2Model::create_param_layers() {
  /*
    内存布局:
        [嵌入层权重] -> pos从这里开始
        [所有层的第一组RMSNorm权重] -> rmsnorm_pos从这里开始
        [所有层的WQ权重] -> pos访问这些
        [所有层的WK权重] -> pos访问这些
        [所有层的WV权重] -> pos访问这些
        [所有层的WO权重] -> pos访问这些
        [所有层的第二组RMSNorm权重] -> rmsnorm_pos跳到这里
        [所有层的W1权重] -> pos访问这些
        [所有层的W2权重] -> pos访问这些
        [所有层的W3权重] -> pos访问这些
        [最终RMSNorm权重] -> rmsnorm_pos最终到达这里
        [分类器权重(如果不共享)] -> pos最终到达这里
  */
  // 函数首先进行两项检查：确保当前不是量化模型, 确保llama_layers_指针已经被初始化
  CHECK(!is_quant_model_);
  CHECK(llama_layers_ != nullptr);
  // The embedding layer
  auto cpu_device_type = base::DeviceType::kDeviceCPU;
  llama_layers_->embedding_layer_ = std::make_shared<op::EmbeddingLayer>(
      device_type_, config_->dim_, config_->seq_len_, std::abs(config_->vocab_size_));
  /*
    从原始模型数据的起始位置(索引0)获取嵌入层权重
    raw_model_data_ 是一个指向预加载模型数据的对象，包含了整个LLaMA2模型的所有权重。
    .weight(0) 方法通过索引0访问存储在模型数据最前面的权重块。
    在LLaMA2的权重布局中，嵌入层(embedding layer)的权重总是位于最前面。
  */
  const void* weight_embedding = raw_model_data_->weight(0);
  // 设置嵌入层的权重，形状为[vocab_size, dim]
  llama_layers_->embedding_layer_->set_weight(0, {std::abs(config_->vocab_size_), config_->dim_},
                                              weight_embedding, cpu_device_type);

  // create all matmul layer
  int32_t dim = config_->dim_;
  // 这里初始化了权重位置追踪器pos，它指向嵌入层权重之后的位置，用于后续加载各层权重。
  size_t pos = dim * std::abs(config_->vocab_size_) + dim * config_->layer_num_;

  // create weight matrix for query 为每一层Transformer创建查询矩阵，维度为[dim, dim]。
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto wq = std::make_shared<op::MatmulLayer>(device_type_, dim, dim);
    wq->set_weight(0, {dim, dim}, this->raw_model_data_->weight(pos), cpu_device_type);
    llama_layers_->wq_layers_.push_back(wq);
    pos += dim * dim;
  }

  // create weight matrix for key
  // 创建键矩阵，注意其维度为[kv_dim_, dim]，这里的kv_dim_可能与主维度dim不同，
  // 这是LLaMA模型的特点之一，用于实现分组查询注意力(GQA)。
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto wk = std::make_shared<op::MatmulLayer>(device_type_, config_->kv_dim_, dim);
    wk->set_weight(0, {config_->kv_dim_, dim}, this->raw_model_data_->weight(pos), cpu_device_type);
    llama_layers_->wk_layers_.push_back(wk);
    pos += config_->kv_dim_ * dim;
  }

  // create weight matrix for value
  // 创建值矩阵，维度同样为[kv_dim_, dim]。
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto wv = std::make_shared<op::MatmulLayer>(device_type_, config_->kv_dim_, dim);
    wv->set_weight(0, {config_->kv_dim_, dim}, this->raw_model_data_->weight(pos), cpu_device_type);
    llama_layers_->wv_layers_.push_back(wv);
    pos += config_->kv_dim_ * dim;
  }

  // create weight matrix for output
  // 创建输出矩阵，维度为[dim, dim]，用于将多头注意力的输出投影回原始维度空间。
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto wo = std::make_shared<op::MatmulLayer>(device_type_, dim, dim);
    wo->set_weight(0, {dim, dim}, this->raw_model_data_->weight(pos), cpu_device_type);
    llama_layers_->wo_layers_.push_back(wo);
    pos += dim * dim;
  }

  // skip ffn rmsnorm 跳过FFN的RMSNorm权重：
  pos += config_->layer_num_ * dim;

  // w1 layers 创建W1层，维度为[hidden_dim, dim]，用于第一次线性变换。
  int32_t hidden_dim = config_->hidden_dim_;
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto w1 = std::make_shared<op::MatmulLayer>(device_type_, hidden_dim, dim);
    w1->set_weight(0, {hidden_dim, dim}, this->raw_model_data_->weight(pos), cpu_device_type);
    llama_layers_->w1_layers_.push_back(w1);
    pos += dim * hidden_dim;
  }

  // w2 layers  创建W2层，维度为[dim, hidden_dim]，用于第二次线性变换。
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto w2 = std::make_shared<op::MatmulLayer>(device_type_, dim, hidden_dim);
    w2->set_weight(0, {dim, hidden_dim}, this->raw_model_data_->weight(pos), cpu_device_type);
    llama_layers_->w2_layers_.push_back(w2);
    pos += dim * hidden_dim;
  }

  // w3 layers 创建W3层，维度为[hidden_dim, dim]，这是LLaMA架构特有的第三个线性变换层，用于实现SwiGLU激活函数。
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto w3 = std::make_shared<op::MatmulLayer>(device_type_, hidden_dim, dim);
    w3->set_weight(0, {hidden_dim, dim}, this->raw_model_data_->weight(pos), cpu_device_type);
    llama_layers_->w3_layers_.push_back(w3);
    pos += dim * hidden_dim;
  }

  // 跳过其他权重
  // skip final rms weight
  pos += dim;
  // skip freqs_cos and freqs_sin weight
  pos += config_->seq_len_ * config_->head_size_;

  // 创建分类层，维度为[vocab_size, dim]，用于最终的token预测。
  llama_layers_->cls_layer_ =
      std::make_shared<op::MatmulLayer>(device_type_, config_->vocab_size_, dim);
  // 这里有一个重要的选择： 如果is_shared_weight_为真，则重用嵌入层的权重(权重共享)
  if (config_->is_shared_weight_) {
    // using token embedding weight
    llama_layers_->cls_layer_->set_weight(0, {config_->vocab_size_, dim},
                                          this->raw_model_data_->weight(0), cpu_device_type);
  } else { // 否则，使用新的独立权重
    llama_layers_->cls_layer_->set_weight(0, {config_->vocab_size_, dim},
                                          this->raw_model_data_->weight(pos), cpu_device_type);
  }

  // create rmsnorm layer 重新初始化权重位置指针rmsnorm_pos来加载RMSNorm层权重。
  /* 这一行设置了rmsnorm_pos的初始值，它指向嵌入层权重之后的位置：
    为什么从这里开始？ 因为在LLaMA2模型中，权重数据的布局是有规律的：
    首先是嵌入层的权重，然后是各种层的归一化权重，
    之后是各层的注意力权重和前馈网络权重。
    这个初始位置指向了第一个RMSNorm层权重的开始位置。
  */
  size_t rmsnorm_pos = config_->dim_ * std::abs(config_->vocab_size_);
  // 创建每个Transformer层中的第一个RMSNorm层，维度为[dim]。
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    std::shared_ptr<op::RmsNormLayer> rms_norm_layer =
        std::make_shared<op::RmsNormLayer>(device_type_, config_->dim_);

    const void* weight_rmsnorm = raw_model_data_->weight(rmsnorm_pos);
    rms_norm_layer->set_weight(0, {config_->dim_}, weight_rmsnorm, cpu_device_type);
    llama_layers_->rmsnorm_layers_.push_back(rms_norm_layer);
    rmsnorm_pos += config_->dim_;
  }
  // 这几行代码跳过了所有层的注意力权重矩阵。 为什么要跳过这些？ 
  // 因为函数已经在前面的部分（通过pos变量）加载了这些权重，这里只是需要跳过它们，以便rmsnorm_pos能指向下一组RMSNorm层的权重。
  // skip attention.wq attention.wk attention.wv attention.wo 跳过注意力相关权重
  rmsnorm_pos += config_->layer_num_ * config_->dim_ * config_->dim_;
  rmsnorm_pos +=
      config_->layer_num_ * config_->dim_ * (config_->kv_head_num_ * config_->head_size_);
  rmsnorm_pos +=
      config_->layer_num_ * config_->dim_ * (config_->kv_head_num_ * config_->head_size_);
  rmsnorm_pos += config_->layer_num_ * config_->dim_ * config_->dim_;
  // 创建每个Transformer层中的第二个RMSNorm层，用于注意力机制之后的归一化。
  // 这个循环为每个Transformer层的第二个RMSNorm层加载权重。
  // 这些RMSNorm层位于注意力机制之后，前馈网络之前，用于再次归一化数据。
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    std::shared_ptr<op::RmsNormLayer> rms_norm_layer =
        std::make_shared<op::RmsNormLayer>(device_type_, config_->dim_);
    const void* weight_rmsnorm = raw_model_data_->weight(rmsnorm_pos);
    rms_norm_layer->set_weight(0, {config_->dim_}, weight_rmsnorm, cpu_device_type);
    llama_layers_->rmsnorm_layers_.push_back(rms_norm_layer);

    rmsnorm_pos += config_->dim_;
  }
  // 这三行跳过了所有层的前馈网络权重矩阵：
  // skip ffn.w1 ffn.w2 ffn.w3 跳过FFN权重
  rmsnorm_pos += config_->layer_num_ * config_->hidden_dim_ * config_->dim_;
  rmsnorm_pos += config_->layer_num_ * config_->hidden_dim_ * config_->dim_;
  rmsnorm_pos += config_->layer_num_ * config_->hidden_dim_ * config_->dim_;

  // 创建网络最后的RMSNorm层，用于最终输出之前的归一化。
  // 经过所有这些调整后，rmsnorm_pos最终指向了整个模型最后的RMSNorm层权重的位置。
  // 这个最终的RMSNorm层应用于整个Transformer堆栈的输出，在进行最终的token预测之前归一化表示。
  std::shared_ptr<op::RmsNormLayer> rms_final_layer =
      std::make_shared<op::RmsNormLayer>(device_type_, config_->dim_);

  const void* weight_rmsnorm_final = raw_model_data_->weight(rmsnorm_pos);
  rms_final_layer->set_weight(0, {config_->dim_}, weight_rmsnorm_final, cpu_device_type);
  llama_layers_->rmsnorm_layers_.push_back(rms_final_layer);
}

// 激活值缓冲区：这才是我们在init_mem()中看到的内存部分，用于存储计算过程中的中间结果和最终输出。
void LLama2Model::init_mem() {
  // 首先根据之前设置的设备类型选择合适的内存分配器
  std::shared_ptr<base::DeviceAllocator> alloc;
  if (device_type_ == base::DeviceType::kDeviceCPU) {
    // 这是一个工厂模式的应用，使得代码能够在不同的硬件环境下透明地工作。
    // 单例模式（get_instance()）确保整个应用程序中只有一个分配器实例，有助于集中管理内存资源。
    alloc = base::CPUDeviceAllocatorFactory::get_instance();
  } else {
    alloc = base::CUDADeviceAllocatorFactory::get_instance();
  }

  // 这是为了确保模型参数在GPU上可用，这对于加速计算至关重要。
  if (device_type_ == base::DeviceType::kDeviceCUDA) {
    CHECK_NE(cuda_config_, nullptr);
    llama_layers_->to_cuda(cuda_config_);
  }
  // 这里同时获取了CPU和CUDA分配器的实例，
  // 这是因为即使在GPU模式下，某些张量（比如输入tokens）仍然需要存储在CPU内存中，而其他张量则需要存储在GPU上以加速计算。
  std::shared_ptr<base::DeviceAllocator> alloc_cpu =
      base::CPUDeviceAllocatorFactory::get_instance();
  std::shared_ptr<base::DeviceAllocator> alloc_cu =
      base::CUDADeviceAllocatorFactory::get_instance();

  // 这些是模型的输入端，input_tokens存储原始的token索引，而input_embeddings则存储这些tokens转换成的向量表示。
  // 存储整数类型的输入token ID，总是分配在CPU上
  tensor::Tensor input_tokens(base::DataType::kDataTypeInt32, 1, true, alloc_cpu);
  // 存储浮点类型的词嵌入向量，维度为模型的嵌入维度（config_->dim_）
  tensor::Tensor input_embeddings(base::DataType::kDataTypeFp32, 1, config_->dim_, true, alloc);

  // 这实现了Transformer模型中的旋转位置编码（RoPE），它帮助模型理解tokens的相对位置
  // 这两个张量用于存储预计算的位置编码值, 大小是注意力头大小乘以序列长度
  tensor::Tensor sin_cache(base::DataType::kDataTypeFp32, config_->head_size_ * config_->seq_len_,
                           true, alloc);
  tensor::Tensor cos_cache(base::DataType::kDataTypeFp32, config_->head_size_ * config_->seq_len_,
                           true, alloc);

  // 这段代码将创建的张量注册到模型的内部缓冲区管理系统中，
  // 每个缓冲区都有一个枚举类型标识符（如kSinCache），使得后续代码可以通过这些标识符检索特定张量。
  // CHECK宏确保每次插入操作都成功。
  CHECK(insert_buffer(ModelBufferType::kSinCache, sin_cache));
  CHECK(insert_buffer(ModelBufferType::kCosCache, cos_cache));
  CHECK(insert_buffer(ModelBufferType::kInputTokens, input_tokens));
  CHECK(insert_buffer(ModelBufferType::kInputEmbeddings, input_embeddings));

  // 这段代码创建并注册了用于存储不同处理阶段中间结果的张量：
  // rms_output：维度为模型的嵌入维度，用于存储RMSNorm（Root Mean Square Normalization）层的输出
  // FIXME:  highlight: 有趣的是，同一个张量被用于多个不同的缓冲区位置，这是一种内存优化技术，因为这些操作是按顺序执行的，可以重用同一块内存。
  tensor::Tensor rms_output(base::DataType::kDataTypeFp32, config_->dim_, true, alloc);
  CHECK(insert_buffer(ModelBufferType::kOutputRMSNorm, rms_output));
  CHECK(insert_buffer(ModelBufferType::kOutputMHA, rms_output));
  CHECK(insert_buffer(ModelBufferType::kW2Output, rms_output));
  CHECK(insert_buffer(ModelBufferType::kFFNRMSNorm, rms_output));

  // 这些张量用于存储前馈神经网络（FFN）层的中间输出
  // 维度为config_->hidden_dim_，通常比模型的主要维度dim_大4倍
  // 在Llama架构中，FFN包含了SwiGLU激活函数，需要W1和W3两个分支的计算结果
  tensor::Tensor w1_output(base::DataType::kDataTypeFp32, config_->hidden_dim_, true, alloc);
  tensor::Tensor w3_output(base::DataType::kDataTypeFp32, config_->hidden_dim_, true, alloc);
  CHECK(insert_buffer(ModelBufferType::kW1Output, w1_output));
  CHECK(insert_buffer(ModelBufferType::kW3Output, w3_output));

  // kv cache 这是自回归生成过程的关键优化部分
  // key_cache和value_cache存储之前生成的每个token的键和值向量
  // 三维张量，维度为[层数, 序列长度, KV维度]
  // 缓存这些值可以避免在生成新token时重复计算之前的键和值，大大提高推理效率
  tensor::Tensor key_cache(base::DataType::kDataTypeFp32, config_->layer_num_, config_->seq_len_,
                           config_->kv_dim_, true, alloc);
  tensor::Tensor value_cache(base::DataType::kDataTypeFp32, config_->layer_num_, config_->seq_len_,
                             config_->kv_dim_, true, alloc);
  CHECK(insert_buffer(ModelBufferType::kKeyCache, key_cache));
  CHECK(insert_buffer(ModelBufferType::kValueCache, value_cache));

  // Wq query output  query：存储注意力机制中的查询向量
  tensor::Tensor query(base::DataType::kDataTypeFp32, config_->dim_, true, alloc);
  CHECK(insert_buffer(ModelBufferType::kQuery, query));

  // Pos tensor  pos_tensor：存储位置索引，用于位置编码
  tensor::Tensor pos_tensor(base::DataType::kDataTypeInt32, 1, true, alloc_cpu);
  CHECK(insert_buffer(ModelBufferType::kInputPos, pos_tensor));

  // Attention output  attn：存储注意力分数，维度为[头数, 序列长度]
  tensor::Tensor attn(base::DataType::kDataTypeFp32, config_->head_num_, config_->seq_len_, true,
                      alloc);
  CHECK(insert_buffer(ModelBufferType::kScoreStorage, attn));
  // FIXME: 注意到kAttnOutput重用了query张量，又一个内存优化实例
  CHECK(insert_buffer(ModelBufferType::kAttnOutput, query));

  // final forward output forward_output：维度为词汇表大小，存储每个可能词元的概率分布
  tensor::Tensor forward_output(base::DataType::kDataTypeFp32, config_->vocab_size_, true, alloc);
  if (device_type_ == base::DeviceType::kDeviceCUDA) {
    // 如果在GPU模式下，还额外创建CPU版本的输出张量，这可能是为了将结果复制回CPU以便进一步处理或采样
    tensor::Tensor forward_output_cpu(base::DataType::kDataTypeFp32, config_->vocab_size_, true,
                                      alloc_cpu);
    CHECK(insert_buffer(ModelBufferType::kForwardOutputCPU, forward_output_cpu));
  }

  CHECK(insert_buffer(ModelBufferType::kForwardOutput, forward_output));
}

base::Status LLama2Model::create_layers() {
  using namespace base;
  if (!llama_layers_) {
    llama_layers_ = std::make_unique<LLama2Layers>();
  }

  if (!is_quant_model_) {
    create_param_layers();
  } else {
    create_param_quant_layers();
  }
  create_nonparam_layers();

  if (!llama_layers_->embedding_layer_) {
    return error::InternalError("Create the embedding layer for the llama model failed!");
  }

  if (llama_layers_->rmsnorm_layers_.size() != 2 * config_->layer_num_ + 1) {
    return error::InternalError("Create the rmsnorm layers for the llama model failed!");
  }

  if (llama_layers_->wq_layers_.size() != config_->layer_num_ ||
      llama_layers_->wk_layers_.size() != config_->layer_num_ ||
      llama_layers_->wv_layers_.size() != config_->layer_num_ ||
      llama_layers_->wo_layers_.size() != config_->layer_num_) {
    return error::InternalError(
        "Create the matmul layer in the attention and ffn attention layers for "
        "the llama model "
        "failed.");
  }

  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    if (!llama_layers_->wq_layers_.at(i) || !llama_layers_->wk_layers_.at(i) ||
        !llama_layers_->wv_layers_.at(i) || !llama_layers_->wo_layers_.at(i)) {
      return error::InternalError(
          "Create the matmul layer in the attention and ffn attention layers for "
          "the llama model "
          "failed.");
    }
  }

  if (llama_layers_->w1_layers_.size() != config_->layer_num_ ||
      llama_layers_->w2_layers_.size() != config_->layer_num_ ||
      llama_layers_->w3_layers_.size() != config_->layer_num_) {
    return error::InternalError(
        "Create the matmul layer in the feedforward layers for the llama model "
        "failed.");
  }

  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    if (!llama_layers_->w1_layers_.at(i) || !llama_layers_->w2_layers_.at(i) ||
        !llama_layers_->w3_layers_.at(i)) {
      return error::InternalError(
          "Create the matmul layer in the feedforward layers for the llama model "
          "failed.");
    }
  }

  if (!llama_layers_->rope_layer_) {
    return error::InternalError("Create the rope layer for the llama model failed!");
  }

  if (!llama_layers_->add_layer_) {
    return error::InternalError("Create the add layer for the llama model failed!");
  }

  if (!llama_layers_->mha_layer_) {
    return error::InternalError("Create the mha layer for the llama model failed!");
  }

  if (!llama_layers_->swiglu_layer_) {
    return error::InternalError("Create the SwiGLU layer for the llama model failed!");
  }
  return error::Success();
}

op::EmbeddingOutput LLama2Model::embedding(const std::vector<int>& tokens) const {
  // 将token ID序列转换为对应的嵌入向量表示。这是模型推理的第一步，将离散的token转换为连续的向量空间表示。

  // 模型使用预分配的缓冲区而不是每次都创建新内存。这些缓冲区在模型初始化时就已分配好，被反复重用以提高效率。
  auto input_tokens = get_buffer(ModelBufferType::kInputTokens);
  auto input_embeddings = get_buffer(ModelBufferType::kInputEmbeddings);
  // 只有在输入大小变化时才重新调整缓冲区形状，这是一个优化措施。
  // 在生成阶段，输入大小通常是固定的（一个token），所以这部分代码可能很少执行。
  if (input_tokens.size() != tokens.size()) {
    input_tokens.reshape({static_cast<int32_t>(tokens.size())});
    input_embeddings.reshape({static_cast<int32_t>(tokens.size()), config_->dim_});
  }
  // 将输入token ID复制到准备好的缓冲区中。
  for (int32_t i = 0; i < tokens.size(); ++i) {
    input_tokens.index<int32_t>(i) = tokens.at(i);
  }

  //  这里创建了一个新张量来记录token数量。注意这是唯一一个在函数内创建的新张量。
  auto input_token_num =
      tensor::Tensor(base::DataType::kDataTypeInt32, static_cast<int32_t>(tokens.size()));
  LOG_IF(FATAL, !llama_layers_->embedding_layer_)
      << "The embedding layer in the llama2 model is null pointer.";
  // 调用嵌入层的前向传播函数，直接在input_embeddings缓冲区内计算结果，不产生额外副本。
  STATUS_CHECK(
      llama_layers_->embedding_layer_->forward(input_tokens, input_token_num, input_embeddings));
  // 将三个组件打包成一个EmbeddingOutput结构并返回。(构造函数中使用了移动语义，所以并没有复制)
  op::EmbeddingOutput output(input_tokens, input_embeddings, input_token_num);
  return output;
}

void LLama2Model::attention_rms(int32_t layer_idx, const tensor::Tensor& input) const {
  CHECK(llama_layers_ != nullptr);
  // attn rmsnorm
  tensor::Tensor rmsnorm_output = get_buffer(ModelBufferType::kOutputRMSNorm);
  std::shared_ptr<op::Layer> rmsnorm_layer = llama_layers_->rmsnorm_layers_.at(layer_idx);
  if (!rmsnorm_layer) {
    LOG(FATAL) << "The attention rmsnorm layer is a null pointer in the llama2 model";
  }
  STATUS_CHECK(rmsnorm_layer->forward(input, rmsnorm_output));
}

void LLama2Model::attention_qkv(int32_t layer_idx, const tensor::Tensor& pos_tensor) const {
  CHECK(llama_layers_ != nullptr);
  // kv cache
  tensor::Tensor query = this->get_buffer(ModelBufferType::kQuery);
  int32_t pos = pos_tensor.index<int32_t>(0);
  // wq wk wv @ input
  const auto& [key, val] = slice_kv_cache(layer_idx, pos);
  // query
  const auto& query_layer = llama_layers_->wq_layers_.at(layer_idx);
  CHECK_NE(query_layer, nullptr) << "The query layer in the attention block is null pointer.";

  auto rmsnorm_output = get_buffer(ModelBufferType::kOutputRMSNorm);
  STATUS_CHECK(query_layer->forward(rmsnorm_output, query));

  // key
  const auto& key_layer = llama_layers_->wk_layers_.at(layer_idx);
  CHECK_NE(key_layer, nullptr) << "The key layer in the attention block is null pointer.";
  STATUS_CHECK(key_layer->forward(rmsnorm_output, key));
  // value
  const auto& value_layer = llama_layers_->wv_layers_.at(layer_idx);
  CHECK_NE(value_layer, nullptr) << "The value layer in the attention block is null pointer.";
  STATUS_CHECK(value_layer->forward(rmsnorm_output, val));

  // rope
  CHECK_NE(llama_layers_->rope_layer_, nullptr)
      << "The RoPE layer in the attention block is null pointer.";
  STATUS_CHECK(llama_layers_->rope_layer_->forward(
      query, key, pos_tensor, get_buffer(ModelBufferType::kSinCache),
      get_buffer(ModelBufferType::kCosCache), tensor::Tensor{}));
}

base::Status LLama2Model::predict(const tensor::Tensor& input, const tensor::Tensor& pos_tensor,
                                  bool is_prompt, int& next) const {
  auto status = forward(input, pos_tensor, next);
  if (!status) {
    return status;
  }
  next = post_processing(pos_tensor, is_prompt);
  return base::error::Success();
}

void LLama2Model::attention_mha(int32_t layer_idx, const tensor::Tensor& pos_tensor) const {
  CHECK(llama_layers_ != nullptr);
  // mha
  tensor::Tensor key_cache = get_buffer(ModelBufferType::kKeyCache);
  // VAL = [val1,val2,...val t]
  // output @ VAL = 最终的结果
  tensor::Tensor val_cache = get_buffer(ModelBufferType::kValueCache);

  tensor::Tensor mha_output = get_buffer(ModelBufferType::kOutputMHA);
  tensor::Tensor score_storage = get_buffer(ModelBufferType::kScoreStorage);
  tensor::Tensor query = this->get_buffer(ModelBufferType::kQuery);

  const auto& mha_layer = llama_layers_->mha_layer_;
  CHECK_NE(mha_layer, nullptr) << "The multi head attention layer is null pointer.";
  int pos = pos_tensor.index<int32_t>(0);
  std::dynamic_pointer_cast<op::MultiHeadAttention>(mha_layer)->set_pos(pos);
  std::dynamic_pointer_cast<op::MultiHeadAttention>(mha_layer)->set_layer_idx(layer_idx);
  STATUS_CHECK(mha_layer->forward(query, score_storage, key_cache, val_cache, mha_output));

  // wo @ attention output
  tensor::Tensor attn_output = get_buffer(ModelBufferType::kAttnOutput);
  const auto& wo_layer = llama_layers_->wo_layers_.at(layer_idx);
  CHECK_NE(wo_layer, nullptr) << "The weight output layer is null pointer.";
  STATUS_CHECK(wo_layer->forward(mha_output, attn_output));
}

void LLama2Model::feed_forward(int32_t layer_idx, const tensor::Tensor& input) const {
  CHECK(llama_layers_ != nullptr);
  // residual add
  CHECK_NE(llama_layers_->add_layer_, nullptr)
      << "The add layer in the feedforward block is null pointer";
  // 首先执行的是残差连接，将输入与注意力输出(kAttnOutput)相加。
  // 这是transformer架构的标准做法，帮助解决深层网络中的梯度消失问题。结果存储回input。
  // 这里最终直接调用add kernel
  STATUS_CHECK(
      llama_layers_->add_layer_->forward(input, get_buffer(ModelBufferType::kAttnOutput), input));

  // 这里使用RMSNorm（均方根归一化）处理前一步的输出。
  // Llama2不使用传统的LayerNorm，而是采用RMSNorm，这是该模型架构的特点之一。
  // 归一化结果存入ffn_norm_output。
  tensor::Tensor ffn_norm_output = get_buffer(ModelBufferType::kFFNRMSNorm);
  const auto& ffn_rmsnorm = llama_layers_->rmsnorm_layers_.at(layer_idx + config_->layer_num_);
  CHECK_NE(ffn_rmsnorm, nullptr)
      << "The final rmsnorm layer in the feedforward block is null pointer";
  STATUS_CHECK(ffn_rmsnorm->forward(input, ffn_norm_output));

  // 这里执行两个并行的线性变换，将归一化后的输入分别通过W1和W3两个权重矩阵。
  // 这是SwiGLU激活函数所需的双路径设计。
  // w1
  tensor::Tensor w1_output = get_buffer(ModelBufferType::kW1Output);
  const auto& w1_layer = llama_layers_->w1_layers_.at(layer_idx);
  CHECK_NE(w1_layer, nullptr) << "The w1 layer in the feedforward block is null pointer";
  STATUS_CHECK(w1_layer->forward(ffn_norm_output, w1_output));

  // w3
  tensor::Tensor w3_ouput = get_buffer(ModelBufferType::kW3Output);
  const auto& w3_layer = llama_layers_->w3_layers_.at(layer_idx);
  CHECK_NE(w3_layer, nullptr) << "The w3 layer in the feedforward block is null pointer";
  STATUS_CHECK(w3_layer->forward(ffn_norm_output, w3_ouput));
  
  // SwiGLU是Llama2使用的门控激活函数，它结合了Swish激活和门控机制, 这一步将W1和W3的输出组合，结果存回w1_output。
  // SwiGLU
  CHECK_NE(llama_layers_->swiglu_layer_, nullptr)
      << "The swiglu layer in the feedforward block is null pointer";
  STATUS_CHECK(llama_layers_->swiglu_layer_->forward(w1_output, w3_ouput, w1_output));
  
  // 将SwiGLU的输出通过最后一个线性层W2进行变换，得到最终的FFN输出。
  // w2
  tensor::Tensor w2_output = get_buffer(ModelBufferType::kW2Output);
  const auto& w2_layer = llama_layers_->w2_layers_.at(layer_idx);
  CHECK_NE(w2_layer, nullptr) << "The w2 layer in the feedforward block is null pointer";
  STATUS_CHECK(w2_layer->forward(w1_output, w2_output));
  
  // 执行最后的残差连接，将原始输入与FFN的输出相加，结果再次存回input，完成整个前馈网络的计算。
  // residual add
  CHECK_NE(llama_layers_->add_layer_, nullptr)
      << "The add layer in the feedforward block is null pointer";
  STATUS_CHECK(llama_layers_->add_layer_->forward(input, w2_output, input));
}

void LLama2Model::cls_logits(const tensor::Tensor& input) const {
  CHECK(llama_layers_ != nullptr);
  const auto& norm = llama_layers_->rmsnorm_layers_.at(2 * config_->layer_num_);
  CHECK_NE(norm, nullptr);
  STATUS_CHECK(norm->forward(input, input));

  tensor::Tensor forward_output = get_buffer(ModelBufferType::kForwardOutput);
  CHECK_NE(llama_layers_->cls_layer_, nullptr);
  STATUS_CHECK(llama_layers_->cls_layer_->forward(input, forward_output));
}

int32_t LLama2Model::post_processing(const tensor::Tensor& pos, bool is_prompt) const {
  tensor::Tensor forward_output = get_buffer(ModelBufferType::kForwardOutput);
  const float* forward_logits = forward_output.ptr<float>();

  int32_t next = 0;
  if (is_prompt) {
    next = -1;
  } else {
    next = static_cast<int32_t>(sampler_->sample(forward_logits, forward_output.size(),
                                                 cuda_config_ ? cuda_config_->stream : nullptr));
  }
  return next;
}

}  // namespace model