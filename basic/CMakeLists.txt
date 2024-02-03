cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(my_tensorrt_project LANGUAGES CXX CUDA)

# 设置C++标准
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_ARCHITECTURES 89)


# 找到CUDA包
find_package(CUDA REQUIRED)

# 包含TensorRT头文件
include_directories("/usr/include/x86_64-linux-gnu")
# 或者，如果你的TensorRT安装在不同的位置，需要指定实际路径
# include_directories("/path/to/your/tensorrt/include")

# 链接TensorRT库
link_directories("/usr/lib/x86_64-linux-gnu")
# 或者，如果你的TensorRT安装在不同的位置，需要指定实际路径
# link_directories("/path/to/your/tensorrt/lib")

# 添加你的源代码文件
add_executable(my_executable 03-bert-infer.cu)

# 链接CUDA库和TensorRT库
target_link_libraries(my_executable nvinfer nvparsers nvonnxparser cudart)
