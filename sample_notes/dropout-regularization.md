# Dropout

**Dropout** randomly zeroes a fraction of activations during training, forcing the
network not to rely on any single unit and acting as regularization. At inference
it is turned off (activations are scaled accordingly). Transformers apply it to
attention weights and sub-layer outputs.
