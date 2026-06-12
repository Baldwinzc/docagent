# Weight Decay

**Weight decay** penalizes large weights, shrinking them slightly each step. It
regularizes the model and improves generalization. With Adam it should be applied
as decoupled decay (AdamW) rather than folded into the gradient as an L2 term.
