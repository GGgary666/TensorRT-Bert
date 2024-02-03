#include <iostream>
#include <iostream>
#include <fstream>
#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <vector>
#include <string>
#include <numeric>
#include <memory>
#include <fstream>
#include <chrono>
using namespace nvinfer1;

std::string model_path = "../../bert-base-uncased/model.plan";

// 创建TensorRT的logger
class Logger : public nvinfer1::ILogger       
{
    void log(Severity severity, const char* msg) noexcept override
    {
        // suppress info-level messages
        if (severity <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
} logger;

size_t getMemorySize(const nvinfer1::Dims& dims, const int32_t elem_size)
{
    return std::accumulate(dims.d, dims.d + dims.nbDims, 1, std::multiplies<int32_t>()) * elem_size;
}

int main() {
    
    // 从文件中读取序列化的模型
    std::ifstream engineFile(model_path, std::ios::binary);
    if (engineFile.fail())
    {
        std::cout << "ERROR!!!!!!!!!!!" << std::endl;
        return 1;
    }
    engineFile.seekg(0, std::ifstream::end);
    auto fsize = engineFile.tellg();
    engineFile.seekg(0, std::ifstream::beg);

    std::vector<char> buffer(fsize);
    engineFile.read(buffer.data(), fsize);
    // 创建runtime
    IRuntime* runtime = createInferRuntime(logger);

    // 反序列化模型
    ICudaEngine* engine = runtime->deserializeCudaEngine(buffer.data(), fsize, nullptr);
    
    // 创建执行上下文
    IExecutionContext* context = engine->createExecutionContext();

    void* buffers[4]; // 假设模型有一个输入和一个输出
    const int input_ids = engine->getBindingIndex("input_ids"); // 根据实际输入层名称替换
    const int attention_mask = engine->getBindingIndex("attention_mask");
    const int token_type_ids = engine->getBindingIndex("token_type_ids");
    const int bert_output = engine->getBindingIndex("logits"); // 根据实际输出层名称替换
    
    auto input_dims = nvinfer1::Dims2{1, 16};
    
    context->setBindingDimensions(input_ids, input_dims);
    context->setBindingDimensions(attention_mask, input_dims);
    context->setBindingDimensions(token_type_ids, input_dims);
    auto output_dims = context->getBindingDimensions(bert_output);
    auto input_size = getMemorySize(input_dims, sizeof(int32_t));
    auto output_size = getMemorySize(output_dims, sizeof(float));
    cudaMalloc(&buffers[input_ids], input_size); // inputSize需要根据模型输入层大小设定
    cudaMalloc(&buffers[attention_mask], input_size);
    cudaMalloc(&buffers[token_type_ids], input_size);
    cudaMalloc(&buffers[bert_output], output_size); // outputSize需要根据模型输出层大小设定
    

    // 准备输入数据到buffers[inputIndex]...
    auto h_input_ids = std::unique_ptr<int>{new int[input_dims.d[1]]};
    auto h_attention_mask = std::unique_ptr<int>{new int[input_dims.d[1]]};
    auto h_token_type_ids = std::unique_ptr<int>{new int[input_dims.d[1]]};
    int input_data[16] = { 101,  1996,  3007,  1997,  2605,  1010,   103,  1010,  3397,  1996,
                                1041, 13355,  2884,  3578,  1012,   102};
    for(int i = 0; i < input_dims.d[1]; i++){
        h_input_ids.get()[i] = input_data[i];
        h_attention_mask.get()[i] = 1;
        h_token_type_ids.get()[i] = 0;

    }

    
    // 将输入数据拷贝至device
    cudaMemcpy(buffers[input_ids], h_input_ids.get(), input_size, cudaMemcpyHostToDevice);
    cudaMemcpy(buffers[attention_mask], h_attention_mask.get(), input_size, cudaMemcpyHostToDevice);
    cudaMemcpy(buffers[token_type_ids], h_token_type_ids.get(), input_size, cudaMemcpyHostToDevice);
    // warmup
    for(int i = 0; i < 5; i++) context->executeV2(buffers);
    // 执行推理
    auto start = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < 10; i++) context->executeV2(buffers);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds> (end - start);
    std::cout << "CPP tensorrt engine with plan model running time (without data movement):"<< duration.count()/10 << " ms" << std::endl;

    // 执行推理
    auto start1= std::chrono::high_resolution_clock::now();
    for(int i = 0; i < 10; i++){
        cudaMemcpy(buffers[input_ids], h_input_ids.get(), input_size, cudaMemcpyHostToDevice);
        cudaMemcpy(buffers[attention_mask], h_attention_mask.get(), input_size, cudaMemcpyHostToDevice);
        cudaMemcpy(buffers[token_type_ids], h_token_type_ids.get(), input_size, cudaMemcpyHostToDevice);
        context->executeV2(buffers);
    } 
    auto end1 = std::chrono::high_resolution_clock::now();
    auto duration1 = std::chrono::duration_cast<std::chrono::milliseconds> (end1 - start1);
    std::cout << "CPP tensorrt engine with plan model running time (with data ovement):"<< duration1.count()/10 << " ms" << std::endl;
    // 从buffers[outputIndex]中获取输出数据，并处理它
    // 注意：需要确保outputBuffer的大小足够大以容纳输出数据
    auto outputBuffer = std::unique_ptr<float>{new float[output_dims.d[1] * output_dims.d[2]]};
    cudaMemcpy(outputBuffer.get(), buffers[bert_output], output_size, cudaMemcpyDeviceToHost);
    // 使用输出数据...
    // C++: 写入二进制文件
    std::ofstream outfile("../../bert-base-uncased/cpp_trt.bin", std::ios::binary);
    outfile.write(reinterpret_cast<char*>(outputBuffer.get()), sizeof(float)*output_dims.d[1] * output_dims.d[2]);
    outfile.close();
    std::cout <<"File has been written to bert-base-uncased/cpp_trt.bin" << std::endl;

    // 清理资源
    cudaFree(buffers[input_ids]);
    cudaFree(buffers[attention_mask]);
    cudaFree(buffers[token_type_ids]);
    cudaFree(buffers[bert_output]);

    return 0;
}