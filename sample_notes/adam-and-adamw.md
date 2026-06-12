# Adam and AdamW

**Adam** adapts each parameter's step using running estimates of the gradient's
mean and variance. **AdamW** fixes how weight decay interacts with that adaptation
by *decoupling* decay from the gradient update, which generalizes better and is the
default optimizer for training Transformers.
