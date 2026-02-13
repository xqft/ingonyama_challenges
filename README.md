# Ingonyama challenges solutions

Some solutions I made for the Ingonyama challenges.

## GPU Long Arithmetic

The objective is to improve a baseline CUDA kernel for extended-precision multiplication of 256-bit integers. They cite [this paper](https://www.acsel-lab.com/arithmetic/arith23/data/1616a047.pdf) stating that the kernel implements section 4. approach: a two-pass algorithm. The two-pass was developed because of a constraint in the Maxwell arch only supporting 16-bit extended multiplication (madc), which produces 16-bit aligned product terms for the row oriented approach that end up taking two 32-bit registers. This problem was solved by NVIDIA on more recent architectures by reintroducing 32-bit extended multiplication.

The kernel actually doesn't implement two-pass as stated, but the row oriented approach using the proper 32-bit madc.

First I tried parallelizing each row product chain into it's own thread, as suggested by the challenge's README. This ended up having the same or worse throughput. I tried using 64-bit PTX instructions with the same result, potentially because these get translated into 32-bit instructions under the hood.

What did work was unrolling a bit the loop at `multVectorsKernel` and instead of multiply->get_u256->multiply->.. changed it to multiply->multiply->get_u256->get_u256 which should help with interleaving and keeping the ALU busy instead of waiting for the 512->256 bit integer conversion before starting the next mul. The optimized kernel finished in almost 1/2 the time of the baseline, so x2 throughput. I then expanded into 4 muls before conversion and the difference improved to 3.2x.

Later I simplified the code a lot by removing the odd/even split and using `mad.wide` which returns the 64 bit result, with the low part immediately being written into the column and the high part stored in an accumulator for the next column. This comes at no change in throughput.

The optimized kernel can be found in `mult_opt.cu`, and the Rust library was adapted to support both kernels for easy bench comparison.