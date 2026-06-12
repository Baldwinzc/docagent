# Quantization

**Quantization** stores weights (and sometimes activations) in lower precision —
8-bit or 4-bit integers instead of 16/32-bit floats — cutting memory and speeding
inference. Post-training quantization is cheapest; quantization-aware training
recovers more accuracy at low bit-widths.
