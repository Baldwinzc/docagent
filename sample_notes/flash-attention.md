# FlashAttention

**FlashAttention** computes exact attention without materializing the full
n x n score matrix in high-bandwidth memory. By tiling the computation and fusing
softmax into the kernel, it is IO-aware — far less memory traffic — giving large
speedups and enabling longer contexts, with identical results to standard attention.
