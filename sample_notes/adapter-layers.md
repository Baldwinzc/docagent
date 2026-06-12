# Adapter Layers

**Adapters** insert small bottleneck modules (down-project, non-linearity,
up-project) inside each Transformer block and train only those while freezing the
base model. One frozen backbone can then serve many tasks by swapping adapters.
