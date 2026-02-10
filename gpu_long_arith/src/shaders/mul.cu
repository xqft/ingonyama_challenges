extern "C" __global__ void inc(unsigned int *out, const unsigned int *in) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < 100) {
        out[i] = in[i] + 1;
    }
}
