# Gradient Accumulation

**Gradient accumulation** sums gradients over several micro-batches before taking
one optimizer step, simulating a large batch size that would not fit in memory. It
trades extra steps for a larger effective batch without more hardware.
