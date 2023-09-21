/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/common/cudaTypeUtils.cuh"
#include "tensorrt_llm/common/reduceKernelUtils.cuh"
#include "tensorrt_llm/kernels/swigluKernels.h"
#include <cuda_runtime.h>
#include <math.h>
using namespace tensorrt_llm::common;

namespace tensorrt_llm
{
namespace kernels
{

template <typename T>
__device__ __forceinline__  T compute_silu(T val)
{
    T one = static_cast<T>(1.f);
    return val / (one + static_cast<T>(expf(-static_cast<float>(val))));
}


template <typename T, int32_t tTPB>
__global__ void swiglu_kernel_func(T* out, const T* input, const int32_t offset)
{

    int32_t indexInput = blockIdx.x * offset * 2 + threadIdx.x;
    int32_t indexOutput = blockIdx.x * offset + threadIdx.x;

    T valueL = 0, valueR = 0; 

#pragma unroll
    for (int32_t i = 0; i < offset / tTPB; ++i)
    {
        valueL = input[indexInput];
        valueR = input[indexInput + offset];
        
        out[indexOutput] =  compute_silu(valueR) * valueL;

        indexInput += tTPB;
        indexOutput += tTPB;
    }
    return;

}

template <typename T>
void invokeSwiglu(T* out, const T* input, const int32_t gridSize, const int32_t nHalfHiddenSize, cudaStream_t stream)
{
    constexpr int32_t blockSize = 256;
    swiglu_kernel_func<T,blockSize><<<gridSize, blockSize, 0, stream>>>(out, input, nHalfHiddenSize);
}


#define INSTANTIATE_SWIGLU(T)                                                                               \
    template void invokeSwiglu(T* out, const T* input, const int32_t gridSize, const int32_t nHalfHiddenSize, cudaStream_t stream);

INSTANTIATE_SWIGLU(float);
INSTANTIATE_SWIGLU(half);

#ifdef ENABLE_BF16
INSTANTIATE_SWIGLU(__nv_bfloat16);
#endif

} // namespace kernels
} // namespace tensorrt_llm
