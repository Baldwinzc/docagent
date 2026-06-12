# Top-k Sampling

**Top-k sampling** restricts sampling to the k most probable tokens (renormalized),
cutting off the long tail of low-probability tokens that cause incoherent text. The
fixed k can be too wide for peaked distributions and too narrow for flat ones.
