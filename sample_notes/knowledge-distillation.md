# Knowledge Distillation

**Distillation** trains a small 'student' model to mimic a large 'teacher', often
by matching the teacher's soft output probabilities (which carry more information
than hard labels). It compresses models for cheaper inference with modest quality
loss (e.g. DistilBERT).
