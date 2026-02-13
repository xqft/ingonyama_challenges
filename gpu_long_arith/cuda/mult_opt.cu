#include <cstdint>
#include <cuda.h>
#include <stdexcept>


namespace ptx {

    __device__ __forceinline__ uint32_t add_cc(const uint32_t x, const uint32_t y) {
        uint32_t result;
        asm volatile("add.cc.u32 %0, %1, %2;" : "=r"(result) : "r"(x), "r"(y));
        return result;
    }

    __device__ __forceinline__ uint32_t addc(const uint32_t x, const uint32_t y) {
        uint32_t result;
        asm volatile("addc.u32 %0, %1, %2;" : "=r"(result) : "r"(x), "r"(y));
        return result;
    }

    __device__ __forceinline__ uint32_t addc_cc(const uint32_t x, const uint32_t y) {
        uint32_t result;
        asm volatile("addc.cc.u32 %0, %1, %2;" : "=r"(result) : "r"(x), "r"(y));
        return result;
    }

    __device__ __forceinline__ uint64_t mad_wide(const uint32_t x, const uint32_t y, const uint64_t z) {
        uint64_t result;
        asm("mad.wide.u32 %0, %1, %2, %3;" : "=l"(result) : "r"(x), "r"(y), "l"(z));
        return result;
    }

} // namespace ptx


struct __align__(16) bigint {
    uint32_t limbs[8];
};

struct __align__(16) bigint_wide {
    uint32_t limbs[16];
};

// stands for "total limbs count"
const int TLC = 8;

static __device__ __forceinline__ void multiply_raw_device(const bigint &as, const bigint &bs, bigint_wide &rs) {
    const uint32_t *a = as.limbs;
    const uint32_t *b = bs.limbs;
    uint32_t cols[TLC * 2] = {0};

    #pragma unroll
    for (int i = 0; i < TLC; i++) {
        uint64_t acc = 0;
        #pragma unroll
        for (int j = 0; j < TLC; j++) {
            acc = ptx::mad_wide(a[j], b[i], acc + cols[i + j]);
            cols[i + j] = (uint32_t)acc;
            acc >>= 32;
        }
        cols[i + TLC] = (uint32_t)acc;
    }

    #pragma unroll
    for (size_t c = 0; c < TLC * 2; c++)
        rs.limbs[c] = cols[c];
}

static __device__ __forceinline__ void add_limbs_device(const uint32_t *x, const uint32_t *y, uint32_t *r) {
    r[0] = ptx::add_cc(x[0], y[0]);
    for (unsigned i = 1; i < (TLC - 1); i++)
        r[i] = ptx::addc_cc(x[i], y[i]);
    r[TLC - 1] = ptx::addc(x[TLC - 1], y[TLC - 1]);
}

// a method to create a 256-bit number from 512-bit result to be able to perpetually
// repeat the multiplication using registers
bigint __device__ __forceinline__ get_256_bit_result(const bigint_wide &xs) {
    const uint32_t *x = xs.limbs;
    bigint out{};
    add_limbs_device(x, &x[TLC], out.limbs);
    return out;
}


// The kernel that does element-wise multiplication of arrays in1 and in2 N times
template <int N>
__global__ void multVectorsKernel(bigint *in1, const bigint *in2, bigint_wide *out, size_t n)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int base = tid * 4;
    if (base + 3 < n)
    {
        bigint a0 = in1[base], a1 = in1[base+1], a2 = in1[base+2], a3 = in1[base+3];
        const bigint b0 = in2[base], b1 = in2[base+1], b2 = in2[base+2], b3 = in2[base+3];
        bigint_wide o0={0}, o1={0}, o2={0}, o3={0};
        for (int i = 0; i < N - 1; i++) {
            multiply_raw_device(a0, b0, o0);
            multiply_raw_device(a1, b1, o1);
            multiply_raw_device(a2, b2, o2);
            multiply_raw_device(a3, b3, o3);
            a0 = get_256_bit_result(o0);
            a1 = get_256_bit_result(o1);
            a2 = get_256_bit_result(o2);
            a3 = get_256_bit_result(o3);
        }
        multiply_raw_device(a0, b0, out[base]);
        multiply_raw_device(a1, b1, out[base+1]);
        multiply_raw_device(a2, b2, out[base+2]);
        multiply_raw_device(a3, b3, out[base+3]);
    }
}

template <int N>
int mult_vectors(bigint in1[], const bigint in2[], bigint_wide *out, size_t n)
{
    // Set the grid and block dimensions
    int threads_per_block = 128;
    int num_blocks = (n / 4 + threads_per_block - 1) / threads_per_block + 1;

    multVectorsKernel<N><<<num_blocks, threads_per_block>>>(in1, in2, out, n);

    return 0;
}


extern "C"
int multiply_test_opt(bigint in1[], const bigint in2[], bigint_wide *out, size_t n)
{
    try
    {
        mult_vectors<1>(in1, in2, out, n);
        return CUDA_SUCCESS;
    }
    catch (const std::runtime_error &ex)
    {
        return -1;
    }
}

extern "C"
int multiply_bench_opt(bigint in1[], const bigint in2[], bigint_wide *out, size_t n)
{
    try
    {
        // for benchmarking, we need to give each thread a number of multiplication tasks that would ensure
        // that we're mostly measuring compute and not global memory accesses, which is why we do 500 multiplications here
        mult_vectors<500>(in1, in2, out, n);
        return CUDA_SUCCESS;
    }
    catch (const std::runtime_error &ex)
    {
        return -1;
    }
}
