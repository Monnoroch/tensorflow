/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#define EIGEN_USE_GPU

#include <algorithm>

#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/kernels/bias_op.h"
#include "tensorflow/core/kernels/bias_op_gpu.h"
#include "tensorflow/core/kernels/reduction_gpu_kernels.cu.h"
#include "tensorflow/core/kernels/reduction_ops_common.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

// There are no native fp16 atomics (we simulate them using 32-bit atomics),
// so fp16 sums are done in fp32 internally. (We don't have a lot of shared
// memory traffic; BiasGradNCHW_SharedAtomics in particular works almost
// entirely on a local variable.)
template <class T>
struct AccumulatorType {
  typedef T type;
};

template <>
struct AccumulatorType<Eigen::half> {
  typedef float type;
};

// Definition of the GPU implementations declared in bias_op.cc.

template <typename T>
__device__ void BiasNHWCKernelImpl(int32 nthreads, const T* __restrict__ input,
                               const T* __restrict__ bias,
                               T* __restrict__ output, int32 bias_size) {
  GPU_1D_KERNEL_LOOP(index, nthreads) {
    int32 bias_offset = index % bias_size;
    output[index] = ldg(input + index) + ldg(bias + bias_offset);
  }
}

template <typename T>
__global__ void BiasNHWCKernel(int32 nthreads, const T* __restrict__ input,
                               const T* __restrict__ bias,
                               T* __restrict__ output, int32 bias_size) {
  BiasNHWCKernelImpl(nthreads, input, bias, output, bias_size);
}


/**
 * This kernel is a versoin of BiasNHWCKernel that uses vetorized memory load / store instructions.
 *
 * In addition to fewer memory fetches the kernel executes more instructions per thread, which means
 * fewer thread blocks can be allocated. Lost parallelism is pretty much always negated by the better
 * memory bandwidth utilization.
 *
 * The cost of the kernel is using more registers (18 vs 13 on my GeForce 1050 Ti).
 *
 * The kernel is only implemented for float -> foat4 pair for testing purposes, but many primitive types
 * can be supported.
 *
 * The kernel is only implemented for bias_size % 4 == 0. It can be extended to support any bias_size
 * but since it's a research project, I'm too lazy to work through the math.
 */
template <typename T>
__global__ void BiasNHWCKernel_biasmod4(int32 nthreads, const T* __restrict__ input,
                               const T* __restrict__ bias,
                               T* __restrict__ output, int32 bias_size) {
  BiasNHWCKernelImpl(nthreads, input, bias, output, bias_size);
}

template <>
__global__ void BiasNHWCKernel_biasmod4(int32 nthreads, const float* __restrict__ input,
                               const float* __restrict__ bias,
                               float* __restrict__ output, int32 bias_size) {
  int threadId = blockIdx.x * blockDim.x + threadIdx.x;
  for (int index = threadId; index < nthreads / 4; index += gridDim.x * blockDim.x) {
    // TODO: pretty sure some kind of math magic might reduce the number of instructions here.
    int32 bias_offset = ((index * 4) % bias_size) / 4;
    float4 bias_vec = ldg(reinterpret_cast<const float4* __restrict__>(bias) + bias_offset);
    float4 input_vec = ldg(reinterpret_cast<const float4* __restrict__>(input) + index);
    reinterpret_cast<float4* __restrict__>(output)[index] = make_float4(
      input_vec.x + bias_vec.x,
      input_vec.y + bias_vec.y,
      input_vec.z + bias_vec.z,
      input_vec.w + bias_vec.w
    );
  }
}

/**
 * This kernel is a version of BiasNHWCKernel that prefetches the bias vector into shared memory.
 * which results in the L1 cache being fully dedicated to cache accesses to the input vector,
 * and hence better memory bandwidth utilization.
 *
 * This kernel can only be used for small bias vectors, less than the shared memory pool size on thr SM,
 * which on my GPU is 48 KiB. Larger shared memory means smaller L1 cache, so in reality the performance gain
 * can only be achieved with small bias vectors, perhaps less than 16 KiB.
 *
 * The kernel is only implemented for float inputs for testing purposes, but all primitive types
 * can be supported.
 *
 * The cost of the kernel is the need for shared memory to fit the bias vector into. It also requires
 * more registers (15 vs 13).
 */
template <typename T>
__global__ void BiasNHWCKernel_sharedmem(int32 nthreads, const T* __restrict__ input,
                               const T* __restrict__ bias,
                               T* __restrict__ output, int32 bias_size) {
  BiasNHWCKernelImpl(nthreads, input, bias, output, bias_size);
}

template <>
__global__ void BiasNHWCKernel_sharedmem(int32 nthreads, const float* __restrict__ input,
                               const float* __restrict__ bias,
                               float* __restrict__ output, int32 bias_size) {
  extern __shared__ float shared_bias[];
  for (int index = threadIdx.x; index < bias_size; index += blockDim.x) {
    shared_bias[index] = ldg(bias + index);
  }
  __syncthreads();

  GPU_1D_KERNEL_LOOP(index, nthreads) {
    int32 bias_offset = index % bias_size;
    __stwt(output + index, __ldcv(input + index) + shared_bias[bias_offset]);
  }
}

/**
 * This kernel is a version of BiasNHWCKernel that combines the optimizations from
 * BiasNHWCKernel_biasmod4 and BiasNHWCKernel_sharedmem. Refer to their documentation
 * for details.
 *
 * This kernel uses 19 registers and requires shared memory for the bias vector.
 *
 * Unfortunately, at least on my GPU, combining these two optimizations does not
 * add any improvement on top of BiasNHWCKernel_biasmod4. It's actually expected:
 * biasmod4 doesn't improve performance by 4x, so we know that the code is no longer
 * limited by memory bandwidth. Therefore, reducing it further by explicitly caching
 * the bias vector instead of relying on automatic L1 cache shouldn't yield any more
 * improvement.
 */
template <typename T>
__global__ void BiasNHWCKernel_biasmod4_sharedmem(int32 nthreads, const T* __restrict__ input,
                               const T* __restrict__ bias,
                               T* __restrict__ output, int32 bias_size) {
  BiasNHWCKernelImpl(nthreads, input, bias, output, bias_size);
}

template <>
__global__ void BiasNHWCKernel_biasmod4_sharedmem(int32 nthreads, const float* __restrict__ input,
                               const float* __restrict__ bias,
                               float* __restrict__ output, int32 bias_size) {
  extern __shared__ float4 shared_bias_vec[];
  for (int index = threadIdx.x; index < bias_size / 4; index += blockDim.x) {
    shared_bias_vec[index] = ldg(reinterpret_cast<const float4* __restrict__>(bias) + index);
  }
  __syncthreads();

  int threadId = blockIdx.x * blockDim.x + threadIdx.x;
  for (int index = threadId; index < nthreads / 4; index += gridDim.x * blockDim.x) {
    // TODO: pretty sure some kind of math magic might reduce the number of instructions here.
    int32 bias_offset = ((index * 4) % bias_size) / 4;
    float4 bias_vec = shared_bias_vec[bias_offset];
    float4 input_vec = __ldcv(reinterpret_cast<const float4* __restrict__>(input) + index);
    __stwt(reinterpret_cast<float4* __restrict__>(output) + index, make_float4(
      input_vec.x + bias_vec.x,
      input_vec.y + bias_vec.y,
      input_vec.z + bias_vec.z,
      input_vec.w + bias_vec.w
    ));
  }
}

template <typename T>
__global__ void BiasNCHWKernel(int32 nthreads, const T* __restrict__ input,
                               const T* __restrict__ bias,
                               T* __restrict__ output, int32 bias_size,
                               int32 image_size) {
  GPU_1D_KERNEL_LOOP(index, nthreads) {
    int32 index2 = index / image_size;
    int32 bias_offset = index2 % bias_size;
    output[index] = ldg(input + index) + ldg(bias + bias_offset);
  }
}

// Add "bias" to "input", broadcasting it on all dimensions but the bias
// dimension.
template <typename T>
void BiasGPU<T>::compute(const GPUDevice& d, const T* input, const T* bias,
                         T* output, int32 batch, int32 height, int32 width,
                         int depth, int32 channel, TensorFormat data_format) {
  const int32 bias_size = channel;
  const int32 image_size = height * width * depth;
  const int32 total_count = batch * bias_size * image_size;
  if (total_count == 0) {
    return;
  }
  GpuLaunchConfig config = GetGpuLaunchConfig(total_count, d);
  if (data_format == FORMAT_NHWC) {
    /**
     * Only run shared memory version for amall bias arrays (<= 16 KiB).
     */
    if (bias_size <= 16384) {
      if (bias_size % 4 == 0) {
        GpuLaunchConfig config = GetGpuLaunchConfig(total_count / 4, d);
        TF_CHECK_OK(GpuLaunchKernel(BiasNHWCKernel_biasmod4_sharedmem<T>, config.block_count,
                                    config.thread_per_block, bias_size * sizeof(T), d.stream(),
                                    total_count, input, bias,
                                    output, bias_size));
      } else {
        TF_CHECK_OK(GpuLaunchKernel(BiasNHWCKernel_sharedmem<T>, config.block_count,
                                    config.thread_per_block, bias_size * sizeof(T), d.stream(),
                                    config.virtual_thread_count, input, bias,
                                    output, bias_size));
      }
    } else {
      /**
       * Only run vectorised version when bias size (and hence the total size) is a multiple of 4.
       */
      if (bias_size % 4 == 0) {
        GpuLaunchConfig config = GetGpuLaunchConfig(total_count / 4, d);
        TF_CHECK_OK(GpuLaunchKernel(BiasNHWCKernel_biasmod4<T>, config.block_count,
                                    config.thread_per_block, 0, d.stream(),
                                    total_count, input, bias,
                                    output, bias_size));
      } else {
        TF_CHECK_OK(GpuLaunchKernel(BiasNHWCKernel<T>, config.block_count,
                                    config.thread_per_block, 0, d.stream(),
                                    config.virtual_thread_count, input, bias,
                                    output, bias_size));
      }
    }
  } else {
    TF_CHECK_OK(GpuLaunchKernel(BiasNCHWKernel<T>, config.block_count,
                                config.thread_per_block, 0, d.stream(),
                                config.virtual_thread_count, input, bias,
                                output, bias_size, image_size));
  }
}

// A naive implementation that is functional on all cases.
template <typename T>
__global__ void BiasGradNHWC_Naive(int32 nthreads,
                                   const T* __restrict__ output_backprop,
                                   T* __restrict__ bias_backprop,
                                   int32 bias_size) {
  GPU_1D_KERNEL_LOOP(index, nthreads) {
    int32 bias_offset = index % bias_size;
    GpuAtomicAdd(bias_backprop + bias_offset, ldg(output_backprop + index));
  }
}

// A naive implementation that is functional on all cases.
template <typename T>
__global__ void BiasGradNCHW_Naive(int32 nthreads,
                                   const T* __restrict__ output_backprop,
                                   T* __restrict__ bias_backprop,
                                   int32 bias_size, int32 image_size) {
  GPU_1D_KERNEL_LOOP(index, nthreads) {
    int32 index2 = index / image_size;
    int32 bias_offset = index2 % bias_size;
    GpuAtomicAdd(bias_backprop + bias_offset, ldg(output_backprop + index));
  }
}

template <typename T>
__global__ void BiasGradNHWC_SharedAtomics(
    int32 nthreads, const T* __restrict__ output_backprop,
    T* __restrict__ bias_backprop, int32 bias_size) {
  typedef typename AccumulatorType<T>::type AccT;
  GPU_DYNAMIC_SHARED_MEM_DECL(8, char, s_buf);
  AccT* s_data = reinterpret_cast<AccT*>(s_buf);
  for (int32 index = threadIdx.x; index < bias_size; index += blockDim.x) {
    s_data[index] = AccT(0);
  }
  __syncthreads();

  for (int32 index = blockIdx.x * blockDim.x + threadIdx.x; index < nthreads;
       index += blockDim.x * gridDim.x) {
    int32 bias_offset = index % bias_size;
    GpuAtomicAdd(s_data + bias_offset, AccT(ldg(output_backprop + index)));
  }
  __syncthreads();

  for (int32 index = threadIdx.x; index < bias_size; index += blockDim.x) {
    GpuAtomicAdd(bias_backprop + index, T(s_data[index]));
  }
}

template <typename T>
__global__ void BiasGradNCHW_SharedAtomics(
    const T* __restrict__ output_backprop, T* __restrict__ bias_backprop,
    int32 batch, int32 bias_size, int32 image_size, int group_size) {
  // Initialize the shared memory.
  typedef typename AccumulatorType<T>::type AccT;
  const int32 kSDataSize = 32;
  __shared__ AccT s_data[kSDataSize];
  for (int32 index = threadIdx.x; index < kSDataSize; index += blockDim.x) {
    s_data[index] = AccT(0);
  }
  __syncthreads();

  // Accumulate all the values within this thread. They all have the same bias
  // index.
  int32 bias_index = blockIdx.x % bias_size;
  int32 group_index = blockIdx.x / bias_size;
  int32 total_count = batch * image_size;
  AccT sum(0);
  for (int32 index = group_index * blockDim.x + threadIdx.x;
       index < total_count; index += blockDim.x * group_size) {
    int32 image_offset = index % image_size;
    int32 batch = index / image_size;
    T val = ldg(output_backprop +
                (batch * bias_size + bias_index) * image_size + image_offset);
    sum += AccT(val);
  }

  // Write the accumulated sum in this thread to the shared memory. Each thread
  // shifts their write location to avoid bank conflict.
  int bias_offset = threadIdx.x % 32;
  GpuAtomicAdd(s_data + bias_offset, sum);
  __syncthreads();

  // Accumulate the results in the shared memory into the first element.
  // No syncthreads is needed since this is only in the same warp.
  int32 thread_index = threadIdx.x;
#if GOOGLE_CUDA
  if (thread_index < 32) {
    AccT data = s_data[thread_index];
    for (int32 delta = warpSize / 2; delta > 0; delta /= 2) {
      data += GpuShuffleXorSync(kCudaWarpAll, data, delta);
    }
    if (thread_index == 0) {
      GpuAtomicAdd(bias_backprop + bias_index, T(data));
    }
  }
#elif TENSORFLOW_USE_ROCM
  if (thread_index < 16) s_data[thread_index] += s_data[thread_index + 16];
  if (thread_index < 8) s_data[thread_index] += s_data[thread_index + 8];
  if (thread_index < 4) s_data[thread_index] += s_data[thread_index + 4];
  if (thread_index < 2) s_data[thread_index] += s_data[thread_index + 2];
  if (thread_index < 1) s_data[thread_index] += s_data[thread_index + 1];

  // The first thread writes out the accumulated result to the global location.
  if (thread_index == 0) {
    GpuAtomicAdd(bias_backprop + bias_index, T(s_data[0]));
  }
#endif
}

template <typename T>
void BiasGradGPU<T>::compute(const GPUDevice& d, const T* output_backprop,
                             T* bias_backprop, int32 batch, int32 height,
                             int32 width, int32 depth, int32 channel,
                             TensorFormat data_format) {
  const int32 bias_size = channel;
  const int32 image_size = height * width * depth;
  const int32 total_count = batch * bias_size * image_size;
  if (total_count == 0) {
    return;
  }
  static constexpr int32 kWarpSize = 32;
  GpuLaunchConfig config = GetGpuLaunchConfig(total_count, d);

  const int max_shared_memory_size = d.sharedMemPerBlock() / 2;
  int32 shared_memory_size = 0;
  if (data_format == FORMAT_NHWC) {
    shared_memory_size = bias_size * sizeof(typename AccumulatorType<T>::type);
  }
  // Check if we have enough shared memory.
  if (shared_memory_size <= max_shared_memory_size) {
    if (data_format == FORMAT_NHWC) {
      TF_CHECK_OK(GpuLaunchKernel(BiasGradNHWC_SharedAtomics<T>,
                                  config.block_count, config.thread_per_block,
                                  shared_memory_size, d.stream(), total_count,
                                  output_backprop, bias_backprop, bias_size));
    } else {
      // Round up the block count to multiple of bias_size.
      int group_size = (config.block_count + bias_size - 1) / bias_size;
      config.block_count = group_size * bias_size;
      if (config.thread_per_block < kWarpSize) {
        config.thread_per_block = kWarpSize;
      }
      TF_CHECK_OK(GpuLaunchKernel(BiasGradNCHW_SharedAtomics<T>,
                                  config.block_count, config.thread_per_block,
                                  0, d.stream(), output_backprop, bias_backprop,
                                  batch, bias_size, image_size, group_size));
    }
  } else {
    // Note that even if we don't have enough shared memory to fit the entire
    // output block, it is possible to process one group of elements at a time.
    // But for now, we simply fall back to the naive implementation.
    if (data_format == FORMAT_NHWC) {
      TF_CHECK_OK(GpuLaunchKernel(
          BiasGradNHWC_Naive<T>, config.block_count, config.thread_per_block, 0,
          d.stream(), total_count, output_backprop, bias_backprop, bias_size));
    } else {
      TF_CHECK_OK(GpuLaunchKernel(BiasGradNCHW_Naive<T>, config.block_count,
                                  config.thread_per_block, 0, d.stream(),
                                  total_count, output_backprop, bias_backprop,
                                  bias_size, image_size));
    }
  }
}

template <typename T>
void BiasGradGPU<T>::DoRowReduction(OpKernelContext* context, T* output,
                                    const T* input, int rows, int cols) {
  typedef const Eigen::array<TTypes<float>::Tensor::Index, 1>& ReductionAxes;
  Constants<GPUDevice> constants;
  gpuprim::Sum op;
  functor::ReduceImpl<T, gpuprim::Sum, T*, const T*, ReductionAxes>(
      context, output, input, 2, rows, cols, 1, 1, constants.kOne, op);
}

template <typename T>
void BiasGradGPU<T>::DoColReduction(OpKernelContext* context, T* output,
                                    const T* input, int rows, int cols) {
  typedef const Eigen::array<TTypes<float>::Tensor::Index, 1>& ReductionAxes;
  Constants<GPUDevice> constants;
  gpuprim::Sum op;
  functor::ReduceImpl<T, gpuprim::Sum, T*, const T*, ReductionAxes>(
      context, output, input, 2, rows, cols, 1, 1, constants.kZero, op);
}

#define DEFINE_GPU_SPECS(T)   \
  template struct BiasGPU<T>; \
  template struct BiasGradGPU<T>;

TF_CALL_GPU_NUMBER_TYPES(DEFINE_GPU_SPECS);

// No BiasGrad kernel for int32.
template struct BiasGPU<int32>;

}  // end namespace tensorflow

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
