#ifndef KUIPER_INCLUDE_MODEL_LLAMA_CONFIG_H_
#define KUIPER_INCLUDE_MODEL_LLAMA_CONFIG_H_
namespace model {
struct ModelConfig {
  int32_t dim = 0;           // 模型维度（embedding维度）
  int32_t hidden_dim = 0;    // 前馈网络隐藏层维度
  int32_t layer_num = 0;     // Transformer层数
  int32_t head_num = 0;      // 注意力头数量
  int32_t kv_head_num = 0;   // Key-Value头数量（用于分组查询注意力GQA）
  int32_t vocab_size = 0;    // 词汇表大小
  int32_t seq_len = 0;       // 序列最大长度
};

struct TransformerConfig {
  int32_t kv_dim_ = 0;        // Key-Value维度
  int32_t kv_mul_ = 0;        // KV倍数（可能用于计算优化）
  int32_t head_size_ = 0;     // 每个注意力头的大小
  int32_t vocab_size_ = 0;    // 词汇表大小
  
  int32_t dim_ = 0;           // 模型维度
  int32_t hidden_dim_ = 0;    // 隐藏层维度
  int32_t layer_num_ = 0;     // 层数
  int32_t head_num_ = 0;      // 注意力头数
  int32_t kv_head_num_ = 0;   // KV头数
  int32_t seq_len_ = 0;       // 序列长度
  
  bool is_shared_weight_ = false;  // 是否共享权重参数
};
}  // namespace model
#endif  // KUIPER_INCLUDE_MODEL_LLAMA_CONFIG_H_
