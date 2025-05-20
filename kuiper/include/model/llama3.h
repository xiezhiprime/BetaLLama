#ifndef KUIPER_INCLUDE_MODEL_LLAMA_H_
#define KUIPER_INCLUDE_MODEL_LLAMA_H_
#include <base/cuda_config.h>
#include "model.h"
#include "op/add.h"
#include "op/embedding.h"
#include "op/rope.h"
#include "op/swiglu.h"
namespace model {

struct LLama2Layers {
  // 1. 单例层(共享层) 每个模型只需一个实例的基础操作层：
  // 执行残差连接中的向量加法 只执行简单的加法运算，无需可学习参数
  std::shared_ptr<op::Layer> add_layer_;
  // 使用三角函数实现位置信息编码 使用确定性数学公式，不需要学习参数
  std::shared_ptr<op::Layer> rope_layer_;
  // 实现SwiGLU激活函数(Swish-Gated Linear Unit)  作为激活函数，只执行非线性变换，不含可学习参数
  std::shared_ptr<op::Layer> swiglu_layer_;
  // 执行注意力计算操作 仅实现计算逻辑，实际权重矩阵(Q/K/V)由其他参数层提供
  std::shared_ptr<op::Layer> mha_layer_;

  // 2. 每个Transformer层特有的层 存储在向量中，每个元素对应一个Transformer层：
  std::vector<std::shared_ptr<op::Layer>> wq_layers_;
  std::vector<std::shared_ptr<op::Layer>> wk_layers_;
  std::vector<std::shared_ptr<op::Layer>> wv_layers_;
  std::vector<std::shared_ptr<op::Layer>> wo_layers_;

  // 前馈网络组件
  std::vector<std::shared_ptr<op::Layer>> w1_layers_;
  std::vector<std::shared_ptr<op::Layer>> w2_layers_;
  std::vector<std::shared_ptr<op::Layer>> rmsnorm_layers_;
  std::vector<std::shared_ptr<op::Layer>> w3_layers_;

  // 3. 输入输出层 处理模型输入输出的专用层：
  std::shared_ptr<op::Layer> cls_layer_;
  std::shared_ptr<op::Layer> embedding_layer_;

  void to_cuda(std::shared_ptr<kernel::CudaConfig> config);
};

class LLama2Model : public Model {
 public:
  explicit LLama2Model(base::TokenizerType tokenizer_type, std::string token_path,
                       std::string model_path, bool is_quant_model);

  base::Status init(base::DeviceType device_type) override;

  base::Status predict(const tensor::Tensor& input, const tensor::Tensor& pos_tensor,
                       bool is_prompt, int& next) const override;

  base::Status forward(const tensor::Tensor& input, const tensor::Tensor& pos_tensor,
                       int& next) const override;

  op::EmbeddingOutput embedding(const std::vector<int>& tokens) const override;

 private:
  void init_mem() override;

  base::Status create_layers() override;

  void create_param_layers() override;

  void create_nonparam_layers() override;

  void create_param_quant_layers() override;

  void attention_mha(int32_t layer_idx, const tensor::Tensor& pos_tensor) const;

  void attention_rms(int32_t layer_idx, const tensor::Tensor& input) const;

  void feed_forward(int32_t layer_idx, const tensor::Tensor& input) const;

  void attention_qkv(int32_t layer_idx, const tensor::Tensor& pos_tensor) const;

  void cls_logits(const tensor::Tensor& input) const;

  int32_t post_processing(const tensor::Tensor& pos, bool is_prompt) const override;

 private:
  std::shared_ptr<kernel::CudaConfig> cuda_config_;
  std::unique_ptr<LLama2Layers> llama_layers_;
};
}  // namespace model

#endif