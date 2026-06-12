# Prefix Tuning

**Prefix tuning** prepends a small set of trainable vectors (a virtual prefix) to
the keys and values at every layer and trains only those, leaving the model frozen.
It steers behaviour with very few parameters.
