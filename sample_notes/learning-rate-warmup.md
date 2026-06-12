# Learning-Rate Warmup

**Warmup** linearly increases the learning rate from near zero over the first few
thousand steps before decaying it. Early in training the gradients are noisy and a
large step can destabilize the model (especially with adaptive optimizers and
layer norm), so warmup is standard for training Transformers.
