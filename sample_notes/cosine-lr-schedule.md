# Cosine Learning-Rate Schedule

After warmup, a **cosine schedule** decays the learning rate following a cosine
curve down to a small final value. It spends more time at higher rates early and
eases into fine adjustments later, and tends to outperform step decay for
large-model training.
