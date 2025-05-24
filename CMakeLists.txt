# 设置最低需要的 CMake 版本为 3.16
cmake_minimum_required(VERSION 3.16)
# 启用编译命令导出，对于代码分析工具如 clang-tidy 很有用
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 明确指定 CUDA 编译器路径为 "/usr/local/cuda/bin/nvcc"
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
# 声明项目使用 C++ 和 CUDA 两种语言
project(llama_infer CXX CUDA)
# 包含自定义的 CUDA 配置文件
include(cmake/cuda.cmake)

# 两者都设置为必需标准，不会自动降级
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# C++ 代码使用 C++17 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
# CUDA 代码使用 CUDA 14 标准
set(CMAKE_CUDA_STANDARD 14)

# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G -g")

# 提供对 LLaMA3 和 Qwen2 模型的支持选项，默认都关闭
# 如果开启，会添加相应的预处理宏定义 -DLLAMA3_SUPPORT 或 -DQWEN2_SUPPORT
option(LLAMA3_SUPPORT OFF)
if (LLAMA3_SUPPORT)
  message(STATUS "LLAMA3 SUPPORT")
  add_definitions(-DLLAMA3_SUPPORT)
endif()
option(QWEN2_SUPPORT OFF)
if (QWEN2_SUPPORT)
  message(STATUS "QWEN2 SUPPORT")
  add_definitions(-DQWEN2_SUPPORT)
endif()
# ---- Add dependencies via CPM ----
# see https://github.com/TheLartians/CPM.cmake for more info
# 提供使用 CPM（CMake Package Manager）进行依赖管理的选项，默认关闭
# 如果启用，将包含多个外部依赖，如 GTest、glog、Armadillo、sentencepiece 等
option(USE_CPM "Use CPM for dependency management" OFF)

if(USE_CPM)
  # Use CPM to manage dependencies
  include(cmake/CPM.cmake)

  CPMAddPackage(
    NAME GTest
    GITHUB_REPOSITORY google/googletest
    VERSION 1.15.0
  )

  CPMAddPackage(
    NAME glog
    GITHUB_REPOSITORY google/glog
    VERSION 0.7.1
    OPTIONS "BUILD_TESTING Off"
  )

  CPMAddPackage(
    NAME Armadillo
    GITLAB_REPOSITORY conradsnicta/armadillo-code
    GIT_TAG 14.0.1
  )

  CPMAddPackage(
    NAME sentencepiece
    GITHUB_REPOSITORY google/sentencepiece
    VERSION 0.2.0
  )
  find_package(sentencepiece REQUIRED)
  # 如果启用 LLaMA3 或 Qwen2 支持，还会额外包含 abseil、re2 和 nlohmann_json 库
  if (LLAMA3_SUPPORT OR QWEN2_SUPPORT)
    CPMAddPackage(
        NAME absl
        GITHUB_REPOSITORY abseil/abseil-cpp
        GIT_TAG 20240722.0
        OPTIONS "BUILD_TESTING Off" "ABSL_PROPAGATE_CXX_STD ON" "ABSL_ENABLE_INSTALL ON"
    )
    CPMAddPackage(
        NAME re2
        GITHUB_REPOSITORY google/re2
        GIT_TAG 2024-07-02
    )
    CPMAddPackage(
        NAME nlohmann_json
        GITHUB_REPOSITORY nlohmann/json
        VERSION 3.11.3
    )
  endif()
endif()

# ---------------------------------
# 确保所需的包可用，找不到会导致构建失败
find_package(GTest REQUIRED)
find_package(glog REQUIRED)
find_package(Armadillo REQUIRED)

# 使用 aux_source_directory 收集各个目录中的源文件，存储到相应的变量中
# 包含了张量操作、基础功能、模型定义、CPU/CUDA 内核等多个模块的源文件
aux_source_directory(kuiper/source/tensor/ DIR_TENSOR)
aux_source_directory(kuiper/source/base/ DIR_BASE)
aux_source_directory(kuiper/source/op/ DIR_OP)
aux_source_directory(kuiper/source/model/ DIR_MODEL)
aux_source_directory(kuiper/source/op/kernels/cpu DIR_KERNEL_CPU)
aux_source_directory(kuiper/source/op/kernels/cuda DIR_KERNEL_CUDA)
aux_source_directory(kuiper/source/op/kernels/ DIR_KERNEL)
aux_source_directory(kuiper/source/sampler DIR_SAMPLE)

# 设置编译后的库文件输出到项目根目录下的 lib 文件夹
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

# 创建名为 "llama" 的共享库，包含所有收集到的源文件
add_library(llama SHARED ${DIR_TENSOR} ${DIR_BASE} ${DIR_OP} ${DIR_KERNEL} ${DIR_MODEL} ${DIR_KERNEL_CPU} ${DIR_KERNEL_CUDA} ${DIR_KERNEL} ${DIR_SAMPLE})
# 链接必要的库，如 sentencepiece、glog、gtest、pthread、CUDA 运行时和 armadillo
target_link_libraries(llama sentencepiece glog::glog gtest gtest_main pthread cudart armadillo)
target_link_directories(llama PUBLIC ${CMAKE_CUDA_COMPILER_LIBRARY_ROOT}/lib64)
# 设置包含目录，确保编译时能找到所需的头文件
target_include_directories(llama PUBLIC ${glog_INCLUDE_DIR})
target_include_directories(llama PUBLIC ${PROJECT_SOURCE_DIR}/kuiper/include)
target_include_directories(llama PUBLIC ${Armadillo_INCLUDE_DIR})
target_include_directories(llama PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

if(USE_CPM)
  # Add sentencepiece include directory
  target_include_directories(llama PUBLIC ${sentencepiece_SOURCE_DIR}/src)
endif()
# 启用 CUDA 可分离编译，允许分别编译 CUDA 代码，有助于提高编译效率
# 保持分离编译
set_target_properties(llama PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# 为Debug模式添加更详细的CUDA调试标志
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G -g -O0 --device-debug")
    # 添加更详细的调试信息
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --generate-line-info")
endif()
# 添加 test 和 demo 子目录到构建系统
add_subdirectory(test)
add_subdirectory(demo)
