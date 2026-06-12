# Special Tokens and Padding

Models reserve **special tokens**: classification/summary tokens (`[CLS]`),
separators (`[SEP]`), beginning/end-of-sequence, and a **padding** token to make
batched sequences equal length. An attention mask tells the model to ignore padded
positions so they do not affect the result.
