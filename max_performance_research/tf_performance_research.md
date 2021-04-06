# Tensorflow performance research

*Disclaimer: This document is a quick-and-dirty draft material for a potential future post on optimising Tensorflow. By no means is it ready or finished, so whoever is reading it please excuse the bad struture, wording or whatever other imperfections it might have.*

[Tensorflow](https://www.tensorflow.org) (a.k.a. TF) is a Machine Learning framework that provides an API for training ML models (mostly parameter-based) and a multi-stage computations execution engine. Rougly speaking:

1. User is writing a program using TF API, usually in Python. That program doesn't perform any computations itself and just creates a large graph-like data structure that is passed to the TF engine
2. First, TF "compiles" the input graph into a chain of "operations". It tries to make it a efficient as possible.
3. Then TF then schedules those operations for execution. They can execute either on a CPU (usually used fir data fetching, etc) or a GPU (usually used for efficient arithmetic).
4. The operations (a.k.a. kernels) run on the devices, computing resutls
5. Results are sent back to the user


Since research can consume any number of compute resources it can get it's hands on, making this pipeline efficient is crucial for achieveing ambitious research golas. By implementing such an architecture Tensorflow provides three ways to improve performance of ML computations:

1. On step (2) the compiler can perform advanced optimizations, such as re-ordering, fusing, unfusing, etc. This document focuses on the two other optimization techniques and this one  won't be discussed
2. On step (3) TF schedules operations on devices (CPU, GPU). Efficient scheduling algorithms can be implemented to fully utilise available compute resources
3. On step (4) devices execute the kernels. Implementing more efficient kernels that have highr hardware awareness speeds up computations

Let's discuss the last topic first.

## Optimising CUDA kernels

As an example for this study I chose the [`BiasAdd`](https://www.tensorflow.org/api_docs/python/tf/nn/bias_add) operation. It's implemented for runinng on GPU with a [`BiasNHWCKernel`](https://github.com/tensorflow/tensorflow/blob/64dde7d21c38b25effd7db944063bfdd20485b5b/tensorflow/core/kernels/bias_op_gpu.cu.cc#L54) CUDA kernel. I chose this kernel for two reasons:

1. It's very simple, literally a single line of logic, so it's both easy to understand and to showcase optimisation techniques on
2. It's implemented as an O(N) algorithm. In ML many tensor operations, such as matrix multiplication are O(N^2) and more, so an O(N) operation probably doesn't get much attention from people optimizing kernels, which provides an opportunity for this research

This is the (slightly simplified) code:
```c++
for (int index = threadId; index < input_vector_size; index += gridDim.x * blockDim.x) {
  int32 bias_offset = index % bias_size;
  output[index] = input[index] + bias[bias_offset];
}
```

It's trivial to understand: each thread just puts a sum of the input value and the bias value for that input into output. The `GPU_1D_KERNEL_LOOP` is needed in case input size is more than the number of threads, then each thread will compute multiple inputs. That's not too important to understand, for the purposes of this post we can think of the GPU as if it has an infinite number of threads.

This code is only executing a few instructions per thread and every thread is essentially working on a single `float` input, plus a `float` bias. It's very fast! Such a simple and obviously efficient code can look like it's impossible to optimise. And indeed, if it was a normal CPU executing it, I would be pressed hard to come up with any optimisations. On a GPU however, there are a few tricks up my sleeve. But before we get to those, let's discuss the first suspect every programmer should turn to when improving performance -- data ... in memory!

### Memory

CPUs nowadays are unbelievably fast: light only travels a few centimeters for each cycle of the CPU clock. CPU executing instructions is rarely a bottleneck in most code. The usual bottleneck is fetching data to perform computations on from memory. It can take hundreds on CPU clock cycles to fetch a few bytes from there. That is why modern CPU designers invent more and more complex cache hierarchies but also branch prediction and prefetching algorithms -- to fight that sloooow RAM latency. To learn more details and numbers, you can (and absolutely should!) read [What Every Programmer Should Know About Memory](https://people.freebsd.org/~lstewart/articles/cpumemory.pdf), it's a masterpiece and despite being written 14 years ago, in 2007, most of it hadn't aged a day.

Now, I said "most code". But people are ingenious, and so we have found a class of algorithms that are, in fact, bound by instruction execution -- large tensor arithmetic. That class can be roughly summarised as "hey, add these two vectors of a billion numbers together". No matter how fast the memory is, adding billions of numbers will take a CPU a fair chunk of time. These algorithms is why [GPGPU platforms](https://en.wikipedia.org/wiki/General-purpose_computing_on_graphics_processing_units) were invented after observing that this is kinda what those GPU [programmable shaders](https://en.wikipedia.org/wiki/Shader) are about -- doing a large number of arthmetic operations that don't depend on each other (i.e. are parallelizable). In fact, even before GPUs that is what [SIMD](https://en.wikipedia.org/wiki/SIMD) instruction sets were about, but these seem to have lost, probably due to a CPUs being increasingly overly complicated to make sequential code efficient, which is something GPUs don't need to optimise for.

However, now that we have GPGPU platforms (and [CUDA](https://en.wikipedia.org/wiki/CUDA) is an implementation of it from NVidia), the computation bottleneck is taken care of and slow memory problem is showing it's ugly head again. Modern GPUs have a much faster DRAM installed, with much higher bandwidth. Still, the processing units are even faster and efficiently utilising both memory bandwidth and latency is critical for best performance.

### Optimising away memory fetches

So, the main performance bottleneck suspect in such a simple algorithm is undoubtedly memory bandwidth. Every memory fetch request is put in the queue and the memory bus has it's own clock to execute those requests sequentially (well, sort of, for the sake of simplicity). GPU multiprocessors (a.k.a. SMs) use a fancy technique of merging memory requests from multiple threads, so if two threads query memory at address `X` and at address `X + 1`, those will only issue a single fetch request because both fetches can be merged into a single fetch over the memory bus. However, because the memory bus is not too wide, fetches for `X` and `X + 20` will likely not be merged and the instruction will be twice as slow because some threads will have to wait for that second fetch to complete. To help the SM to fetch memory efficiently we can batch them manually, thus reducing the number of fetches and increasing the probability of fusion. In this example I am fetching 4 floats at a time instead of one by reading the builtin `float4` type instead of `float`:
```c++
int threadId = blockIdx.x * blockDim.x + threadIdx.x;
for (int index = threadId; index < input_vector_size / 4; index += gridDim.x * blockDim.x) {
  int32 bias_offset = ((index * 4) % bias_size) / 4;
  float4 bias_vec = reinterpret_cast<const float4* __restrict__>(bias)[bias_offset];
  float4 input_vec = reinterpret_cast<const float4* __restrict__>(input)[index];
  reinterpret_cast<float4* __restrict__>(output)[index] = make_float4(
    input_vec.x + bias_vec.x,
    input_vec.y + bias_vec.y,
    input_vec.z + bias_vec.z,
    input_vec.w + bias_vec.w
  );
}
```

This code:
- Performs 4 times fewer memory fetches
- That are each 4 times bigger
- Executes more instructions per thread: 4 floating adds instead of 1 and some aditional bias offset and loop integer math
- And so it requires more registers (18 vs 13 on my GPU)
- And it only supports bias vectors that have their `size % 4 == 0`. I could fix this one by spending some time working through the modulo `bias_size` arithmetics
- Additionally, since this algorithm processes 4 inputs per thread, we also run 4 times fewer thread blocks and free up compute resources for other concurrent operations to utilize

So, does it perform better? Indeed it does! According to the Tensorflow profiler we are saving 9% execution time.


| ![Baseline from the repository](baseline.png?raw=true "Baseline from the repository") |
|:---------:|
| Fig. 1. Baseline kernel from the repository |


| ![With batced fetches](batched_fetch.png?raw=true "With batced fetches") |
|:---------:|
| Fig. 2. Kernel with batched fetches |


Results so far:

Algorithm | Execution time, ns | Relative execution time
:----------:|:--------------------:|:------------------------:
 Baseline | 47,416,329 | 1.0
 Batched fetch | 43,151,414 | 0.91 (-9%)


Okay, so can be improve it even further by fetching even more floats? Well, maybe, but not on my [GeForce GTX 1050 TI](https://www.nvidia.com/en-sg/geforce/products/10series/geforce-gtx-1050) GPU. It has a 128 bit memory bus size, which means it can fetch exactly 4 floats at a time. So, on my machine increasing the amount of fetched memory per thread would only reduce parallelism and therefore hurt performance.

### Cache-friendly code

Okay, so we are utilizing memory bandwidth better. But every fetch is still a very slow operation that each thread has to wait for while doing nothing. Can we somehow speed those up? Turns out yes, we can -- by using a memory cache!

Modern NVida GPUs have a [hierarchy](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#architecture-5-x) of memory units:

0. Global DRAM memory
1. Global L2 cache that all SMs use to cache reads
3. Local per-SM L1 cache, very local @ much fast!
2. Per-SM texture cache, which is just like L1 cache but read-only, so it doesn't have a concept of coherence
4. Per-SM shared memory, which is essentially just L1 cache that the programmer has direct access to instead of it being automatic
5. Per-thread registers, the fastest of them all

Generally speaking we want to move all interactions with memory down this hierarchy. Let's look at the code again:

```c++
for (int index = threadId; index < input_vector_size; index += gridDim.x * blockDim.x) {
  int32 bias_offset = index % bias_size;
  output[index] = input[index] + bias[bias_offset];
}
```

It has:

1. A read of `input + index`
2. A read of `bias + bias_offset`
3. A write to `output + index`

Note that the `bias` vector is usually small, because it's a single `tf.Variable`, while the `input` vector is usually large because it's a minibatch of values, but also because it may have more dimensions than the bias, for example when a vector bias is added to every pixel of a multi-channel image. So `input` reads keep flushing `bias` out of the L1 cache (actually, texture cache because the original code uses [`ldg()`](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#ldg-function) to load data). To prefent that, we can reserve a portion of L1 cache for `bias`. That is called "shared memory". The algorithm is as follows:

1. Populate shared memory with the `bias` vector. That needs to be done per-block, and each thread in a block will only populate a small portion of the data for maximum parallelism
2. Wait for all threads to finish populating the shared memory with `__syncthreads()`
3. Run the existing code, but reading bias from the shared memory instead of global one

```c++
extern __shared__ float shared_bias[];
for (int index = threadIdx.x; index < bias_size; index += blockDim.x) {
  shared_bias[index] = bias[index];
}
__syncthreads();

for (int index = threadId; index < input_vector_size; index += gridDim.x * blockDim.x) {
  int32 bias_offset = index % bias_size;
  output[index] = input[index] + shared_bias[bias_offset];
}
```

Also note that each input read (1) and output write (3) are only executed once. Not once per thread or SM. Just once. So we don't need them to use caches, neither for reading, not for writing. We can help the compiler to understand it by using [special instructions](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#cache-operators) that skip the caching subsystem:

```c++
__stwt(output + index, __ldcv(input + index) + shared_bias[bias_offset]);
```

Those optimizations result in a 4% performance improvement. The number is so low because the `input` vector is 2^14 times larger than the bias vector in my experiments, so fetching the bias is happening over and over, so the caching subsystem probably doesn't evict it most of the time. Still it proves this technique is a viable one.

| ![With explicitly cached bias](shared_memory.png?raw=true "With explicitly cached bias") |
|:---------:|
| Fig. 3. Kernel with explicitly cached bias |

Results so far:

Algorithm | Execution time, ns | Relative execution time
:----------:|:--------------------:|:------------------------:
 Baseline | 47,416,329 | 1.0
 Batched fetch | 43,151,414 | 0.91 (-9%)
 Explicit bias cache | 45,615,811 | 0.96 (-4%)

I actually did measure those two improvements independently and the second change does add a small, but statistically significant speedup, thought not one worth discussing separately here.

### Let's combine those together!

Now that we have two successfull optimizations, let's combine tham and see what happens.

But first, let's try to predict the results. In this case I think the second optimization will only provide a marginal improvement over the first one. The reason it'll only be marginal comes from another, more advanced concept -- [memory banks](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory-5-x). Shared memory consists of banks organized so that each consecutive 32-bit word (so, one `float`) is stored in the next memory bank. And when we read a `float4` we are querying 4 memory banks at once, which means 4 threads query a single bank concurrently. Those 4 reads must be seialized, therefore the shared memory read is 4 times slower than it could be. Still, I expect a small improvement, since reading shared memory is much mre reliable than reading the automatic L1 cache. I could go further and make code memory-bank-aware, but I would like to leave it as an excercise to the reader.

The final code is:

```c++
extern __shared__ float4 shared_bias_vec[];
for (int index = threadIdx.x; index < bias_size / 4; index += blockDim.x) {
  shared_bias_vec[index] = ldg(reinterpret_cast<const float4* __restrict__>(bias) + index);
}
__syncthreads();

int threadId = blockIdx.x * blockDim.x + threadIdx.x;
for (int index = threadId; index < input_vector_size / 4; index += gridDim.x * blockDim.x) {
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
```

And the result confirms my intuition -- it's an 11% performance improvement over the baseline, which is an additional 2% over the first optimization, not a 4% as one might naively expect!

| ![With batched fetch and explicit bias cache](batched_fetch_shared_mem.png?raw=true "With batched fetch and explicit bias cache") |
|:---------:|
| Fig. 4. Kernel with batched fetch and explicit bias cache |

Results:

Algorithm | Execution time, ns | Relative execution time
:----------:|:--------------------:|:------------------------:
 Baseline | 47,416,329 | 1.0
 Batched fetch | 43,151,414 | 0.91 (-9%)
 Explicit bias cache | 45,615,811 | 0.96 (-4%)
 Batched fetch and explicit bias cache | 42,238,542 | 0.89 (-11%)

### Conclusion

On modern hardware memory access is the cornerstone of performance, which is especially true for parallel computing. *Incidentally, it's also true for parallel batch computing systems such as [MapReduce](https://en.wikipedia.org/wiki/MapReduce), except there it's not RAM, but HDD performance that's usually the bottleneck.*

I explored a two techniques to optimise memory access patterns that proved useful even for a seemingly trivial one-line algorithm. I also proposed a follow up improvement of making code memory-bank-aware.

Now, let's move on to the second topic.

## Scheduling

*Normally would be a second post, but I don't have enough material and I'm not sure I'll have the time to work on it more.*

Let's discuss the scheduling. Unfortunately Tensorflow profiler doesn't really help with this, so I had to use [NVidia Visual Profiler](https://developer.nvidia.com/nvidia-visual-profiler). I generated 2 epochs worth of profiling data on the baseline kernel with my example code.

The code is organized as follows:

1. The graph is as simple as possible, no trainable variables or gradients, just pure forward computation
2. The NN consists of the MNIST-shaped input layer and 30 `ManyBiasAdd` layers
3. Each `ManyBiasAdd` layes is a sum of adding 30 different biases to the input

So we get 900 `BiasAdd` and 30 `AddN` operations in a single batch run. Logically, I would expect:

a. 30 `BiasAdd` operations to be executed in parallel (my GPU has 4GiB of DRAM)
b. `AddN` to be executed to add those together
c. GOTO (a) for the next layer

However, that is not what I observe in the profiler. In my case the 30 `BiasAdd` operations were always executed sequentially:

| ![NVVP profiling results](nvvm_profile_batch.png?raw=true "NVVP profiling results") |
|:---------:|
| Fig. 5. NVVP profiling results |

Closer inspections tells me that all of the `BiasAdd` operations were executed on a single "Stream 14". That is why they are sequential.

As a result I believe the Tensorflow scheduler could be significantly improved by allowing to run independend kernels in parallel.
