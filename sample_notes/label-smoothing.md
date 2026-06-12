# Label Smoothing

**Label smoothing** replaces hard one-hot targets with a distribution that puts a
small mass on the wrong classes. It discourages the model from becoming
over-confident, improves calibration, and often slightly improves accuracy; the
original Transformer used it for translation.
