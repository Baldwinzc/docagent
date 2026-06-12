# Temperature Sampling

**Temperature** rescales the logits before the softmax: <1 sharpens the
distribution (more deterministic), >1 flattens it (more random/creative). It is the
basic knob trading off coherence against diversity in sampling-based decoding.
