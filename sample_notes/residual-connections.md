# Residual Connections

Each Transformer sub-layer is wrapped in a **residual (skip) connection**: the
block computes `x + Sublayer(x)` rather than `Sublayer(x)`. Adding the input back
gives gradients a direct path to earlier layers, which makes very deep stacks
trainable and lets a layer learn a small refinement instead of a full remapping.
Residuals are paired with layer normalization in every Transformer block.
