#include <base/base.h>
#include <base/tick.h>
#include <glog/logging.h>
#include "model/llama3.h"

/*
  返回值：int32_t - 实际执行的生成步数

  参数：
    model：LLama2模型的常量引用
    sentence：输入提示文本
    total_steps：最大生成步数限制
    need_output：是否需要输出生成结果（默认为false）
*/
int32_t generate(const model::LLama2Model& model, const std::string& sentence, int total_steps,
                 bool need_output = false) {
  // 这部分将输入文本编码为token序列，并检查是否为空。
  auto tokens = model.encode(sentence);
  int32_t prompt_len = tokens.size();
  LOG_IF(FATAL, tokens.empty()) << "The tokens is empty.";

  int32_t pos = 0; // pos：当前处理的位置
  // 在下面的处理过程中，next一直在发生变化，即将转到生成阶段的时候，其指向的就是prompt的最后一个token，
  int32_t next = -1; // next：下一个预测的token  
  bool is_prompt = true; // is_prompt：标记当前是否处于提示词处理阶段
  const auto& prompt_embedding = model.embedding(tokens); // prompt_embedding：提示词的嵌入表示
  tensor::Tensor pos_tensor = model.get_buffer(model::ModelBufferType::kInputPos); // pos_tensor：位置张量  

  std::vector<int32_t> words; // words：存储生成的token序列
  while (pos < total_steps) {
    // 在语言模型中，这行代码的作用是：更新位置张量，告诉模型当前正在处理序列中的哪个位置。
    pos_tensor.index<int32_t>(0) = pos;
    if (pos < prompt_len - 1) {
      /*
        提示词字符串 → tokens向量[t₁,t₂,...,tₙ] → embedding_output(包含所有token的嵌入) →
                    → fill_input(选择第pos个嵌入) → model.predict → 继续处理下一个位置
      */
      tensor::Tensor input = model.fill_input(pos_tensor, prompt_embedding, is_prompt);
      model.predict(input, pos_tensor, is_prompt, next);
    } else {
      /*
        前一步生成的token(next) → 单元素向量[next] → token_embedding(包含单个token的嵌入) →
        → fill_input(选择唯一的嵌入，索引=0) → model.predict → 生成新的next → 循环继续
      */
      is_prompt = false;
      // 创建一个只包含最新token的全新向量 ： 将单个token ID转换成只有一个元素的向量
      tokens = std::vector<int32_t>{next};
      // 只获取这个token的嵌入
      const auto& token_embedding = model.embedding(tokens);
      // input : (config_->dim_, )
      tensor::Tensor input = model.fill_input(pos_tensor, token_embedding, is_prompt);
      model.predict(input, pos_tensor, is_prompt, next);
    }
    if (model.is_sentence_ending(next)) {
      break;
    }
    if (is_prompt) {
      // 这部分使用提示词的嵌入来预测模型的状态，但实际上是在"预热"模型，下一个token直接从提示词中获取：
      next = tokens.at(pos + 1);
      words.push_back(next);
    } else {
      // 在这个阶段，函数使用先前预测的token来生成下一个token，然后添加到结果中：
      words.push_back(next);
    }

    pos += 1;
  }
  if (need_output) {
    // 如果设置了need_output，函数会将生成的token解码为文本并打印出来。最后返回实际执行的步数。
    printf("%s ", model.decode(words).data());
    fflush(stdout);
  }
  return std::min(pos, total_steps);
}


int main(int argc, char* argv[]) {
  if (argc != 3) {
    LOG(INFO) << "Usage: ./demo checkpoint path tokenizer path";
    return -1;
  }
  const char* checkpoint_path = argv[1];  // e.g. out/model.bin
  const char* tokenizer_path = argv[2];

  // 创建了一个 LLama2Model 类的实例，使用 SPE 编码类型的分词器
  model::LLama2Model model(base::TokenizerType::kEncodeSpe, tokenizer_path,
    checkpoint_path, false);
  auto init_status = model.init(base::DeviceType::kDeviceCUDA);
  // 检查初始化是否成功，如果失败则输出错误代码并终止程序
  if (!init_status) {
    LOG(FATAL) << "The model init failed, the error code is: " << init_status.get_err_code();
  }
  const std::string& sentence = "hello";

  auto start = std::chrono::steady_clock::now();
  printf("Generating...\n");
  fflush(stdout);
  int steps = generate(model, sentence, 128, true);
  auto end = std::chrono::steady_clock::now();
  auto duration = std::chrono::duration<double>(end - start).count();
  printf("\nsteps/s:%lf\n", static_cast<double>(steps) / duration);
  fflush(stdout);
  return 0;
}
