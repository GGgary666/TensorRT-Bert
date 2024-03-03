/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
 
#include "LayerNormPlugin.h"
#include <iostream>
#include <cstdint>
#include <cuda_fp16.h>
using namespace nvinfer1;
// typedef unsigned short half;

PluginFieldCollection LayerNormPluginCreator::fc_{};
std::vector<PluginField> LayerNormPluginCreator::attr_;


template <typename T>
__global__ void layerNormKernel(T *pInput, T *pOutput)
{
    const int tx = threadIdx.x, index = blockIdx.x * 768 + threadIdx.x;

    __shared__ T temp[128];
    // 这里会不会越界
    T value0 = pInput[index];
    T value1 = pInput[index + 128];
    T value2 = pInput[index + 256];
    T value3 = pInput[index + 384];
    T value4 = pInput[index + 512];
    T value5 = pInput[index + 640];
    temp[tx] = value0 + value1 + value2 + value3 + value4 + value5;
    __syncthreads();

    for (int stride = 64; stride >= 1; stride /= 2)
    {
        if (tx < stride)
        {
            temp[tx] += temp[tx + stride];
        }
        __syncthreads();
    }
    T mean = temp[0] / (T) 768.0;
    __syncthreads();

    temp[tx] = (value0 - mean) * (value0 - mean) + (value1 - mean) * (value1 - mean) + (value2 - mean) * (value2 - mean) +
               (value3 - mean) * (value3 - mean) + (value4 - mean) * (value4 - mean) + (value5 - mean) * (value5 - mean);
    __syncthreads();

    for (int stride = 64; stride >= 1; stride /= 2)
    {
        if (tx < stride)
        {
            temp[tx] += temp[tx + stride];
        }
        __syncthreads();
    }
    T var = temp[0] / (T) 768.0;
    T eps = 6e-6;
    pOutput[index]       = (value0 - mean) * (T) rsqrtf(var + eps);
    pOutput[index + 128] = (value1 - mean) * (T) rsqrtf(var + eps);
    pOutput[index + 256] = (value2 - mean) * (T) rsqrtf(var + eps);
    pOutput[index + 384] = (value3 - mean) * (T) rsqrtf(var + eps);
    pOutput[index + 512] = (value4 - mean) * (T) rsqrtf(var + eps);
    pOutput[index + 640] = (value5 - mean) * (T) rsqrtf(var + eps);
}

void layerNormCompute(const int nBlock, cudaStream_t stream, const float* input, float* output)
{
    layerNormKernel<float> <<<nBlock, 128, 0, stream>>>((float *)input, (float *)output);
}

void layerNormCompute(const int nBlock, cudaStream_t stream, const __half* input, __half* output)
{
    layerNormKernel<__half> <<<nBlock, 128, 0, stream>>>((__half *)input, (__half *)output);
}

int32_t LayerNormPlugin::enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    const int nBlock = inputDesc[0].dims.d[0] * inputDesc[0].dims.d[1];
    // const int dim = inputDesc[0].dims.d[inputDesc[0].dims.nbDims - 1];

    // cast float to half
    // const __half* input = static_cast<const __half*>(inputs[0]);
    // __half* output = static_cast<__half*>(outputs[0]);

    const float* input = static_cast<const float*>(inputs[0]);
    float* output = static_cast<float*>(outputs[0]);


    layerNormCompute(nBlock, stream, input, output);
    // layerNormKernel<float> <<<nBlock, 128, 0, stream>>>((float *)inputs[0], (float *)outputs[0]);
    return 0;
}

REGISTER_TENSORRT_PLUGIN(LayerNormPluginCreator);


// #include "LayerNormPlugin.h"
// #include <iostream>
// using namespace nvinfer1;

// PluginFieldCollection LayerNormPluginCreator::fc_{};
// std::vector<PluginField> LayerNormPluginCreator::attr_;


// __global__ void layerNormKernel(float *pInput, float *pOutput)
// {
//     const int tx = threadIdx.x, index = blockIdx.x * 768 + threadIdx.x;

//     __shared__ float temp[128];
//     // 这里会不会越界
//     float value0 = pInput[index];
//     float value1 = pInput[index + 128];
//     float value2 = pInput[index + 256];
//     float value3 = pInput[index + 384];
//     float value4 = pInput[index + 512];
//     float value5 = pInput[index + 640];
//     temp[tx] = value0 + value1 + value2 + value3 + value4 + value5;
//     __syncthreads();

//     for (int stride = 64; stride >= 1; stride /= 2)
//     {
//         if (tx < stride)
//         {
//             temp[tx] += temp[tx + stride];
//         }
//         __syncthreads();
//     }
//     float mean = temp[0] / 768;
//     __syncthreads();

//     temp[tx] = (value0 - mean) * (value0 - mean) + (value1 - mean) * (value1 - mean) + (value2 - mean) * (value2 - mean) +
//                (value3 - mean) * (value3 - mean) + (value4 - mean) * (value4 - mean) + (value5 - mean) * (value5 - mean);
//     __syncthreads();

//     for (int stride = 64; stride >= 1; stride /= 2)
//     {
//         if (tx < stride)
//         {
//             temp[tx] += temp[tx + stride];
//         }
//         __syncthreads();
//     }
//     float var = temp[0] / 768;

//     pOutput[index]       = (value0 - mean) * rsqrtf(var + 6e-6);
//     pOutput[index + 128] = (value1 - mean) * rsqrtf(var + 6e-6);
//     pOutput[index + 256] = (value2 - mean) * rsqrtf(var + 6e-6);
//     pOutput[index + 384] = (value3 - mean) * rsqrtf(var + 6e-6);
//     pOutput[index + 512] = (value4 - mean) * rsqrtf(var + 6e-6);
//     pOutput[index + 640] = (value5 - mean) * rsqrtf(var + 6e-6);
// }


// int32_t LayerNormPlugin::enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
// {
//     const int nBlock = inputDesc[0].dims.d[0] * inputDesc[0].dims.d[1];
//     // const int dim = inputDesc[0].dims.d[inputDesc[0].dims.nbDims - 1];

        
//     layerNormKernel <<<nBlock, 128, 0, stream>>>((float *)inputs[0], (float *)outputs[0]);
//     return 0;
// }

// REGISTER_TENSORRT_PLUGIN(LayerNormPluginCreator);