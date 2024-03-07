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

const int NUM_THREADS = 768 / 4;
const int WARP_SIZE = 32;
const int K = 768;

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int mask = WARP_SIZE >> 1; mask >= 1; mask >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}

__device__ __forceinline__ float block_reduce_sum(float val) {
  // always <= 32 warps per block (limited by 1024 threads per block)
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    int warp = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    static __shared__ float shared[NUM_WARPS];

    val = warp_reduce_sum(val);
    if (lane == 0) shared[warp] = val;
    __syncthreads();
    val = (lane < NUM_WARPS) ? shared[lane] : 0.0f;
    val = warp_reduce_sum(val);
    return val;
}

#define FLOAT4(value) (reinterpret_cast<float4*>(&(value))[0])
template <typename T>
__global__ void layerNormKernel(T *x, T* g, T* b, T *y) {
    int tid = threadIdx.x; // 0..K-1
    int bid = blockIdx.x; // 0..N-1
    int idx = (bid * blockDim.x + threadIdx.x) * 4;
    const float epsilon = 6e-6f;

    __shared__ float s_mean; // shared within block
    __shared__ float s_variance; // shared within block
    float4 reg_x = FLOAT4(x[idx]);
    float value = reg_x.x + reg_x.y + reg_x.z + reg_x.w;
    float sum = block_reduce_sum(value);
    if (tid == 0) s_mean = sum / (float) K;
    // wait for s_mean in shared memory to be ready for all threads
    __syncthreads();
    float4 reg_x_hat;
    reg_x_hat.x = reg_x.x - s_mean;
    reg_x_hat.y = reg_x.y - s_mean;
    reg_x_hat.z = reg_x.z - s_mean;
    reg_x_hat.w = reg_x.w - s_mean;
    float variance = reg_x_hat.x * reg_x_hat.x + reg_x_hat.y * reg_x_hat.y
                    + reg_x_hat.z * reg_x_hat.z + reg_x_hat.w * reg_x_hat.w;
    variance = block_reduce_sum(variance);
    if (tid == 0) s_variance = rsqrtf(variance / (float) K + epsilon);
    // wait for s_variance in shared memory to be ready for all threads
    __syncthreads();
    float4 reg_y;
    reg_y.x = reg_x_hat.x * s_variance * g[tid * 4] + b[tid * 4];
    reg_y.y = reg_x_hat.y * s_variance * g[tid * 4 + 1] + b[tid * 4 + 1];
    reg_y.z = reg_x_hat.z * s_variance * g[tid * 4 + 2] + b[tid * 4 + 2];
    reg_y.w = reg_x_hat.w * s_variance * g[tid * 4 + 3] + b[tid * 4 + 3];
    FLOAT4(y[idx]) = reg_y;
}


// template <typename T>
// __device__ __forceinline__ float warp_reduce_sum(T val)
// {
// #pragma unroll
//     for (int mask = WARP_SIZE >> 1; mask >= 1; mask >>= 1)
//     {
//         val += __shfl_xor_sync(0xffffffff, val, mask);
//     }
//     return val;
// }

// template <typename T>
// __device__ __forceinline__ float block_reduce_sum(T val)
// {
//     // always <= 32 warps per block (limited by 1024 threads per block)
//     constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
//     int warp = threadIdx.x / WARP_SIZE;
//     int lane = threadIdx.x % WARP_SIZE;
//     static __shared__ T shared[NUM_WARPS];

//     val = warp_reduce_sum<T>(val);
//     if (lane == 0)
//         shared[warp] = val;
//     __syncthreads();
//     val = (lane < NUM_WARPS) ? shared[lane] : (T)0.0;
//     val = warp_reduce_sum<T>(val);
//     return val;
// }

// template <typename T>
// __global__ void layerNormKernel(T *x, T *g, T *b, T *y)
// {
//     int tid = threadIdx.x; // 0..K-1
//     int bid = blockIdx.x;  // 0..N-1
//     int idx = bid * blockDim.x + threadIdx.x;
//     const T epsilon = (T)6e-6;

//     __shared__ T s_mean;     // shared within block
//     __shared__ T s_variance; // shared within block
//     T value = x[idx];        // load once only
//     T gamma = g[tid];
//     T beta = b[tid];
//     T sum = block_reduce_sum<T>(value);
//     if (tid == 0)
//         s_mean = sum / (T)K;
//     // wait for s_mean in shared memory to be ready for all threads
//     __syncthreads();
//     T variance = (value - s_mean) * (value - s_mean);
//     variance = block_reduce_sum<T>(variance);
//     if (tid == 0)
//         s_variance = rsqrtf(variance / (T)K + epsilon);
//     // wait for s_variance in shared memory to be ready for all threads
//     __syncthreads();
//     // y[idx] = (value - s_mean) * s_variance;

//     y[idx] =  (value - s_mean) * s_variance * gamma + beta;

    
// }

// template <typename T>
// __global__ void layerNormKernel(T *pInput, T *pOutput)
// {
//     const int tx = threadIdx.x, index = blockIdx.x * 768 + threadIdx.x;

//     __shared__ T temp[128];
//     // 这里会不会越界
//     T value0 = pInput[index];
//     T value1 = pInput[index + 128];
//     T value2 = pInput[index + 256];
//     T value3 = pInput[index + 384];
//     T value4 = pInput[index + 512];
//     T value5 = pInput[index + 640];
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
//     T mean = temp[0] / (T) 768.0;
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
//     T var = temp[0] / (T) 768.0;
//     T eps = 6e-6;
//     pOutput[index]       = (value0 - mean) * (T) rsqrtf(var + eps);
//     pOutput[index + 128] = (value1 - mean) * (T) rsqrtf(var + eps);
//     pOutput[index + 256] = (value2 - mean) * (T) rsqrtf(var + eps);
//     pOutput[index + 384] = (value3 - mean) * (T) rsqrtf(var + eps);
//     pOutput[index + 512] = (value4 - mean) * (T) rsqrtf(var + eps);
//     pOutput[index + 640] = (value5 - mean) * (T) rsqrtf(var + eps);
// }

void layerNormCompute(const int nBlock, cudaStream_t stream, const float *input, const float *g, const float *b, float *output)
{
    layerNormKernel<float><<<nBlock, NUM_THREADS, 0, stream>>>((float *)input, (float *)g, (float *)b, (float *)output);
}
// void layerNormCompute(const int nBlock, cudaStream_t stream, const float* input, float* output)
// {
//     layerNormKernel<float> <<<nBlock, NUM_THREADS, 0, stream>>>((float*)input, (float*)output);
// }
// void layerNormCompute(const int nBlock, cudaStream_t stream, const __half* input, __half* output)
// {
//     layerNormKernel<__half> <<<nBlock, NUM_THREADS, 0, stream>>>((__half *)input, (__half *)output);
// }

int32_t LayerNormPlugin::enqueue(const PluginTensorDesc *inputDesc, const PluginTensorDesc *outputDesc, const void *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream) noexcept
{
    const int nBlock = inputDesc[0].dims.d[0] * inputDesc[0].dims.d[1];
    // const int dim = inputDesc[0].dims.d[inputDesc[0].dims.nbDims - 1];

    // cast float to half
    // const __half* input = static_cast<const __half*>(inputs[0]);
    // __half* output = static_cast<__half*>(outputs[0]);

    const float *input = static_cast<const float *>(inputs[0]);
    const float *gamma_ptr = static_cast<const float *>(inputs[1]);
    const float *beta_ptr = static_cast<const float *>(inputs[2]);
    float *output = static_cast<float *>(outputs[0]);

    // layerNormCompute(nBlock, stream, input, output);
    layerNormCompute(nBlock, stream, input, gamma_ptr, beta_ptr, output);
    return 0;
}

REGISTER_TENSORRT_PLUGIN(LayerNormPluginCreator);































// /*
//  * Copyright (c) 2019-2021, NVIDIA CORPORATION.  All rights reserved.
//  *
//  * Licensed under the Apache License, Version 2.0 (the "License");
//  * you may not use this file except in compliance with the License.
//  * You may obtain a copy of the License at
//  *
//  *     http://www.apache.org/licenses/LICENSE-2.0
//  *
//  * Unless required by applicable law or agreed to in writing, software
//  * distributed under the License is distributed on an "AS IS" BASIS,
//  * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  * See the License for the specific language governing permissions and
//  * limitations under the License.
//  */
 
// #include "LayerNormPlugin.h"
// #include <iostream>
// #include <cstdint>
// #include <cuda_fp16.h>
// using namespace nvinfer1;
// // typedef unsigned short half;

// PluginFieldCollection LayerNormPluginCreator::fc_{};
// std::vector<PluginField> LayerNormPluginCreator::attr_;

// const int NUM_THREADS = 768 / 4;
// const int WARP_SIZE = 32;
// const int K  = 768;


// __device__ __forceinline__ float warp_reduce_sum(float val) {
//     #pragma unroll
//     for (int mask = WARP_SIZE >> 1; mask >= 1; mask >>= 1) {
//         val += __shfl_xor_sync(0xffffffff, val, mask);
//     }
//     return val;
// }

// __device__ __forceinline__ float block_reduce_sum(float val) {
//   // always <= 32 warps per block (limited by 1024 threads per block)
//     constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
//     int warp = threadIdx.x / WARP_SIZE;
//     int lane = threadIdx.x % WARP_SIZE;
//     static __shared__ float shared[NUM_WARPS];
    
//     val = warp_reduce_sum(val);
//     if (lane == 0) shared[warp] = val;
//     __syncthreads();
//     val = (lane < NUM_WARPS) ? shared[lane] : 0.0f;
//     val = warp_reduce_sum(val);
//     return val;
// }


// #define FLOAT4(value) (reinterpret_cast<float4*>(&(value))[0])
// template <typename T> 
// __global__ void layerNormKernel(T *x, T *y) {
//     int tid = threadIdx.x; // 0..K-1
//     int bid = blockIdx.x; // 0..N-1
//     int idx = (bid * blockDim.x + threadIdx.x) * 4;
//     const float epsilon = 6e-6f;

//     __shared__ float s_mean; // shared within block
//     __shared__ float s_variance; // shared within block
//     float4 reg_x = FLOAT4(x[idx]);
//     float value = reg_x.x + reg_x.y + reg_x.z + reg_x.w;
//     float sum = block_reduce_sum(value);
//     if (tid == 0) s_mean = sum / (float) K;
//     // wait for s_mean in shared memory to be ready for all threads
//     __syncthreads();
//     float4 reg_x_hat;
//     reg_x_hat.x = reg_x.x - s_mean;
//     reg_x_hat.y = reg_x.y - s_mean;
//     reg_x_hat.z = reg_x.z - s_mean;
//     reg_x_hat.w = reg_x.w - s_mean;
//     float variance = reg_x_hat.x * reg_x_hat.x + reg_x_hat.y * reg_x_hat.y 
//                     + reg_x_hat.z * reg_x_hat.z + reg_x_hat.w * reg_x_hat.w;
//     variance = block_reduce_sum(variance);/*
//  * Copyright (c) 2019-2021, NVIDIA CORPORATION.  All rights reserved.
//  *
//  * Licensed under the Apache License, Version 2.0 (the "License");
//  * you may not use this file except in compliance with the License.
//  * You may obtain a copy of the License at
//  *
//  *     http://www.apache.org/licenses/LICENSE-2.0
//  *
//  * Unless required by applicable law or agreed to in writing, software
//  * distributed under the License is distributed on an "AS IS" BASIS,
//  * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  * See the License for the specific language governing permissions and
//  * limitations under the License.
//  */
 
// #include "LayerNormPlugin.h"
// #include <iostream>
// #include <cstdint>
// #include <cuda_fp16.h>
// using namespace nvinfer1;
// // typedef unsigned short half;

// PluginFieldCollection LayerNormPluginCreator::fc_{};
// std::vector<PluginField> LayerNormPluginCreator::attr_;

// const int NUM_THREADS = 768 / 4;
// const int WARP_SIZE = 32;
// const int K  = 768;


// __device__ __forceinline__ float warp_reduce_sum(float val) {
//     #pragma unroll
//     for (int mask = WARP_SIZE >> 1; mask >= 1; mask >>= 1) {
//         val += __shfl_xor_sync(0xffffffff, val, mask);
//     }
//     return val;
// }

// __device__ __forceinline__ float block_reduce_sum(float val) {
//   // always <= 32 warps per block (limited by 1024 threads per block)
//     constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
//     int warp = threadIdx.x / WARP_SIZE;
//     int lane = threadIdx.x % WARP_SIZE;
//     static __shared__ float shared[NUM_WARPS];
    
//     val = warp_reduce_sum(val);
//     if (lane == 0) shared[warp] = val;
//     __syncthreads();
//     val = (lane < NUM_WARPS) ? shared[lane] : 0.0f;
//     val = warp_reduce_sum(val);
//     return val;
// }


// #define FLOAT4(value) (reinterpret_cast<float4*>(&(value))[0])
// template <typename T> 
// __global__ void layerNormKernel(T *x, T *y) {
//     int tid = threadIdx.x; // 0..K-1
//     int bid = blockIdx.x; // 0..N-1
//     int idx = (bid * blockDim.x + threadIdx.x) * 4;
//     const float epsilon = 6e-6f;

//     __shared__ float s_mean; // shared within block
//     __shared__ float s_variance; // shared within block
//     float4 reg_x = FLOAT4(x[idx]);
//     float value = reg_x.x + reg_x.y + reg_x.z + reg_x.w;
//     float sum = block_reduce_sum(value);
//     if (tid == 0) s_mean = sum / (float) K;
//     // wait for s_mean in shared memory to be ready for all threads
//     __syncthreads();
//     float4 reg_x_hat;
//     reg_x_hat.x = reg_x.x - s_mean;
//     reg_x_hat.y = reg_x.y - s_mean;
//     reg_x_hat.z = reg_x.z - s_mean;
//     reg_x_hat.w = reg_x.w - s_mean;
//     float variance = reg_x_hat.x * reg_x_hat.x + reg_x_hat.y * reg_x_hat.y 
//                     + reg_x_hat.z * reg_x_hat.z + reg_x_hat.w * reg_x_hat.w;
//     variance = block_reduce_sum(variance);
//     if (tid == 0) s_variance = rsqrtf(variance / (float) K + epsilon);
//     // wait for s_variance in shared memory to be ready for all threads
//     __syncthreads();
//     float4 reg_y;
//     reg_y.x = reg_x_hat.x * s_variance;
//     reg_y.y = reg_x_hat.y * s_variance;
//     reg_y.z = reg_x_hat.z * s_variance;
//     reg_y.w = reg_x_hat.w * s_variance;
//     FLOAT4(y[idx]) = reg_y;
// }

// // template <typename T> 
// // __device__ __forceinline__ float warp_reduce_sum(T val) {
// //     #pragma unroll
// //     for (int mask = WARP_SIZE >> 1; mask >= 1; mask >>= 1) {
// //         val += __shfl_xor_sync(0xffffffff, val, mask);
// //     }
// //     return val;
// // }

// // template <typename T> 
// // __device__ __forceinline__ float block_reduce_sum(T val) {
// //   // always <= 32 warps per block (limited by 1024 threads per block)
// //     constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
// //     int warp = threadIdx.x / WARP_SIZE;
// //     int lane = threadIdx.x % WARP_SIZE;
// //     static __shared__ T shared[NUM_WARPS];
    
// //     val = warp_reduce_sum<T>(val);
// //     if (lane == 0) shared[warp] = val;
// //     __syncthreads();
// //     val = (lane < NUM_WARPS) ? shared[lane] : (T) 0.0;
// //     val = warp_reduce_sum<T>(val);
// //     return val;
// // }


// // template <typename T> 
// // __global__ void layerNormKernel(T *x, T *y)
// // {
// //     int tid = threadIdx.x; // 0..K-1
// //     int bid = blockIdx.x; // 0..N-1
// //     int idx = bid * blockDim.x + threadIdx.x;
// //     const T epsilon = (T) 6e-6;

// //     __shared__ T s_mean; // shared within block
// //     __shared__ T s_variance; // shared within block
// //     T value =  x[idx] ; // load once only
// //     T sum = block_reduce_sum<T>(value);
// //     if (tid == 0) s_mean = sum / (T) K;
// //     // wait for s_mean in shared memory to be ready for all threads
// //     __syncthreads();
// //     T variance = (value - s_mean) * (value - s_mean);
// //     variance = block_reduce_sum<T>(variance);
// //     if (tid == 0) s_variance = rsqrtf(variance / (T) K + epsilon);
// //     // wait for s_variance in shared memory to be ready for all threads
// //     __syncthreads();
// //     y[idx] = (value - s_mean) * s_variance;

// // }


// // template <typename T>
// // __global__ void layerNormKernel(T *pInput, T *pOutput)
// // {
// //     const int tx = threadIdx.x, index = blockIdx.x * 768 + threadIdx.x;

// //     __shared__ T temp[128];
// //     // 这里会不会越界
// //     T value0 = pInput[index];
// //     T value1 = pInput[index + 128];
// //     T value2 = pInput[index + 256];
// //     T value3 = pInput[index + 384];
// //     T value4 = pInput[index + 512];
// //     T value5 = pInput[index + 640];
// //     temp[tx] = value0 + value1 + value2 + value3 + value4 + value5;
// //     __syncthreads();

// //     for (int stride = 64; stride >= 1; stride /= 2)
// //     {
// //         if (tx < stride)
// //         {
// //             temp[tx] += temp[tx + stride];
// //         }
// //         __syncthreads();
// //     }
// //     T mean = temp[0] / (T) 768.0;
// //     __syncthreads();

// //     temp[tx] = (value0 - mean) * (value0 - mean) + (value1 - mean) * (value1 - mean) + (value2 - mean) * (value2 - mean) +
// //                (value3 - mean) * (value3 - mean) + (value4 - mean) * (value4 - mean) + (value5 - mean) * (value5 - mean);
// //     __syncthreads();

// //     for (int stride = 64; stride >= 1; stride /= 2)
// //     {
// //         if (tx < stride)
// //         {
// //             temp[tx] += temp[tx + stride];
// //         }
// //         __syncthreads();
// //     }
// //     T var = temp[0] / (T) 768.0;
// //     T eps = 6e-6;
// //     pOutput[index]       = (value0 - mean) * (T) rsqrtf(var + eps);
// //     pOutput[index + 128] = (value1 - mean) * (T) rsqrtf(var + eps);
// //     pOutput[index + 256] = (value2 - mean) * (T) rsqrtf(var + eps);
// //     pOutput[index + 384] = (value3 - mean) * (T) rsqrtf(var + eps);
// //     pOutput[index + 512] = (value4 - mean) * (T) rsqrtf(var + eps);
// //     pOutput[index + 640] = (value5 - mean) * (T) rsqrtf(var + eps);
// // }

// void layerNormCompute(const int nBlock, cudaStream_t stream, const float* input, float* output)
// {
//     layerNormKernel<float> <<<nBlock, NUM_THREADS, 0, stream>>>((float *)input, (float *)output);
// }

// void layerNormCompute(const int nBlock, cudaStream_t stream, const __half* input, __half* output)
// {
//     layerNormKernel<__half> <<<nBlock, NUM_THREADS, 0, stream>>>((__half *)input, (__half *)output);
// }

// int32_t LayerNormPlugin::enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
// {
//     const int nBlock = inputDesc[0].dims.d[0] * inputDesc[0].dims.d[1];
//     // const int dim = inputDesc[0].dims.d[inputDesc[0].dims.nbDims - 1];

//     // cast float to half
//     // const __half* input = static_cast<const __half*>(inputs[0]);
//     // __half* output = static_cast<__half*>(outputs[0]);

//     const float* input = static_cast<const float*>(inputs[0]);
//     float* output = static_cast<float*>(outputs[0]);


//     layerNormCompute(nBlock, stream, input, output);
//     // layerNormKernel<float> <<<nBlock, 128, 0, stream>>>((float *)inputs[0], (float *)outputs[0]);
//     return 0;
// }

// REGISTER_TENSORRT_PLUGIN(LayerNormPluginCreator);


//     if (tid == 0) s_variance = rsqrtf(variance / (float) K + epsilon);
//     // wait for s_variance in shared memory to be ready for all threads
//     __syncthreads();
//     float4 reg_y;
//     reg_y.x = reg_x_hat.x * s_variance;
//     reg_y.y = reg_x_hat.y * s_variance;
//     reg_y.z = reg_x_hat.z * s_variance;
//     reg_y.w = reg_x_hat.w * s_variance;
//     FLOAT4(y[idx]) = reg_y;
// }

// // template <typename T> 
// // __device__ __forceinline__ float warp_reduce_sum(T val) {
// //     #pragma unroll
// //     for (int mask = WARP_SIZE >> 1; mask >= 1; mask >>= 1) {
// //         val += __shfl_xor_sync(0xffffffff, val, mask);
// //     }
// //     return val;
// // }

// // template <typename T> 
// // __device__ __forceinline__ float block_reduce_sum(T val) {
// //   // always <= 32 warps per block (limited by 1024 threads per block)
// //     constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
// //     int warp = threadIdx.x / WARP_SIZE;
// //     int lane = threadIdx.x % WARP_SIZE;
// //     static __shared__ T shared[NUM_WARPS];
    
// //     val = warp_reduce_sum<T>(val);
// //     if (lane == 0) shared[warp] = val;
// //     __syncthreads();
// //     val = (lane < NUM_WARPS) ? shared[lane] : (T) 0.0;
// //     val = warp_reduce_sum<T>(val);
// //     return val;
// // }


// // template <typename T> 
// // __global__ void layerNormKernel(T *x, T *y)
// // {
// //     int tid = threadIdx.x; // 0..K-1
// //     int bid = blockIdx.x; // 0..N-1
// //     int idx = bid * blockDim.x + threadIdx.x;
// //     const T epsilon = (T) 6e-6;

// //     __shared__ T s_mean; // shared within block
// //     __shared__ T s_variance; // shared within block
// //     T value =  x[idx] ; // load once only
// //     T sum = block_reduce_sum<T>(value);
// //     if (tid == 0) s_mean = sum / (T) K;
// //     // wait for s_mean in shared memory to be ready for all threads
// //     __syncthreads();
// //     T variance = (value - s_mean) * (value - s_mean);
// //     variance = block_reduce_sum<T>(variance);
// //     if (tid == 0) s_variance = rsqrtf(variance / (T) K + epsilon);
// //     // wait for s_variance in shared memory to be ready for all threads
// //     __syncthreads();
// //     y[idx] = (value - s_mean) * s_variance;

// // }


// // template <typename T>
// // __global__ void layerNormKernel(T *pInput, T *pOutput)
// // {
// //     const int tx = threadIdx.x, index = blockIdx.x * 768 + threadIdx.x;

// //     __shared__ T temp[128];
// //     // 这里会不会越界
// //     T value0 = pInput[index];
// //     T value1 = pInput[index + 128];
// //     T value2 = pInput[index + 256];
// //     T value3 = pInput[index + 384];
// //     T value4 = pInput[index + 512];
// //     T value5 = pInput[index + 640];
// //     temp[tx] = value0 + value1 + value2 + value3 + value4 + value5;
// //     __syncthreads();

// //     for (int stride = 64; stride >= 1; stride /= 2)
// //     {
// //         if (tx < stride)
// //         {
// //             temp[tx] += temp[tx + stride];
// //         }
// //         __syncthreads();
// //     }
// //     T mean = temp[0] / (T) 768.0;
// //     __syncthreads();

// //     temp[tx] = (value0 - mean) * (value0 - mean) + (value1 - mean) * (value1 - mean) + (value2 - mean) * (value2 - mean) +
// //                (value3 - mean) * (value3 - mean) + (value4 - mean) * (value4 - mean) + (value5 - mean) * (value5 - mean);
// //     __syncthreads();

// //     for (int stride = 64; stride >= 1; stride /= 2)
// //     {
// //         if (tx < stride)
// //         {
// //             temp[tx] += temp[tx + stride];
// //         }
// //         __syncthreads();
// //     }
// //     T var = temp[0] / (T) 768.0;
// //     T eps = 6e-6;
// //     pOutput[index]       = (value0 - mean) * (T) rsqrtf(var + eps);
// //     pOutput[index + 128] = (value1 - mean) * (T) rsqrtf(var + eps);
// //     pOutput[index + 256] = (value2 - mean) * (T) rsqrtf(var + eps);
// //     pOutput[index + 384] = (value3 - mean) * (T) rsqrtf(var + eps);
// //     pOutput[index + 512] = (value4 - mean) * (T) rsqrtf(var + eps);
// //     pOutput[index + 640] = (value5 - mean) * (T) rsqrtf(var + eps);
// // }

// void layerNormCompute(const int nBlock, cudaStream_t stream, const float* input, float* output)
// {
//     layerNormKernel<float> <<<nBlock, NUM_THREADS, 0, stream>>>((float *)input, (float *)output);
// }

// void layerNormCompute(const int nBlock, cudaStream_t stream, const __half* input, __half* output)
// {
//     layerNormKernel<__half> <<<nBlock, NUM_THREADS, 0, stream>>>((__half *)input, (__half *)output);
// }

// int32_t LayerNormPlugin::enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
// {
//     const int nBlock = inputDesc[0].dims.d[0] * inputDesc[0].dims.d[1];
//     // const int dim = inputDesc[0].dims.d[inputDesc[0].dims.nbDims - 1];

//     // cast float to half
//     // const __half* input = static_cast<const __half*>(inputs[0]);
//     // __half* output = static_cast<__half*>(outputs[0]);

//     const float* input = static_cast<const float*>(inputs[0]);
//     float* output = static_cast<float*>(outputs[0]);


//     layerNormCompute(nBlock, stream, input, output);
//     // layerNormKernel<float> <<<nBlock, 128, 0, stream>>>((float *)inputs[0], (float *)outputs[0]);
//     return 0;
// }

// REGISTER_TENSORRT_PLUGIN(LayerNormPluginCreator);

