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

    __device__ __forceinline__ uint32_t mul_lo(const uint32_t x, const uint32_t y) {
        uint32_t result;
        asm("mul.lo.u32 %0, %1, %2;" : "=r"(result) : "r"(x), "r"(y));
        return result;
    }

    __device__ __forceinline__ uint32_t mul_hi(const uint32_t x, const uint32_t y) {
        uint32_t result;
        asm("mul.hi.u32 %0, %1, %2;" : "=r"(result) : "r"(x), "r"(y));
        return result;
    }

    __device__ __forceinline__ uint32_t mad_lo_cc(const uint32_t x, const uint32_t y, const uint32_t z) {
        uint32_t result;
        asm volatile("mad.lo.cc.u32 %0, %1, %2, %3;" : "=r"(result) : "r"(x), "r"(y), "r"(z));
        return result;
    }

    __device__ __forceinline__ uint32_t mad_hi_cc(const uint32_t x, const uint32_t y, const uint32_t z) {
        uint32_t result;
        asm volatile("mad.hi.cc.u32 %0, %1, %2, %3;" : "=r"(result) : "r"(x), "r"(y), "r"(z));
        return result;
    }

    __device__ __forceinline__ uint32_t madc_lo_cc(const uint32_t x, const uint32_t y, const uint32_t z) {
        uint32_t result;
        asm volatile("madc.lo.cc.u32 %0, %1, %2, %3;" : "=r"(result) : "r"(x), "r"(y), "r"(z));
        return result;
    }

    __device__ __forceinline__ uint32_t madc_hi_cc(const uint32_t x, const uint32_t y, const uint32_t z) {
        uint32_t result;
        asm volatile("madc.hi.cc.u32 %0, %1, %2, %3;" : "=r"(result) : "r"(x), "r"(y), "r"(z));
        return result;
    }

    __device__ __forceinline__ uint32_t madc_hi(const uint32_t x, const uint32_t y, const uint32_t z) {
        uint32_t result;
        asm volatile("madc.hi.u32 %0, %1, %2, %3;" : "=r"(result) : "r"(x), "r"(y), "r"(z));
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
        // low terms
        cols[i] = ptx::mad_lo_cc(a[0], b[i], cols[i]);
        #pragma unroll
        for (size_t j = 1; j < TLC; j++)
            cols[i + j] = ptx::madc_lo_cc(a[j], b[i], cols[i + j]);
        cols[i + TLC] = ptx::addc(cols[i + TLC], 0);

        // high terms
        cols[i + 1] = ptx::mad_hi_cc(a[0], b[i], cols[i + 1]);
        #pragma unroll
        for (size_t j = 1; j < TLC - 1; j++)
            cols[i + j + 1] = ptx::madc_hi_cc(a[j], b[i], cols[i + j + 1]);
        cols[i + TLC] = ptx::madc_hi(a[TLC - 1], b[i], cols[i + TLC]);
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
    if (tid < n)
    {
        bigint i1 = in1[tid];
        const bigint i2 = in2[tid];
        bigint_wide o = {0};
        for (int i = 0; i < N - 1; i++) {
            multiply_raw_device(i1, i2, o);
            i1 = get_256_bit_result(o);
        }
        multiply_raw_device(i1, i2, out[tid]);
    }
}

template <int N>
int mult_vectors(bigint in1[], const bigint in2[], bigint_wide *out, size_t n)
{
    // Set the grid and block dimensions
    int threads_per_block = 128;
    int num_blocks = (n + threads_per_block - 1) / threads_per_block + 1;

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
