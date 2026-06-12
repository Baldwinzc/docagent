# Reinforcement Learning from Human Feedback (RLHF)

**RLHF** aligns a model with human preferences: collect human comparisons of
outputs, train a **reward model** to predict them, then optimize the language model
against that reward (often with PPO) plus a penalty for drifting too far from the
base model. It improves helpfulness and safety beyond plain instruction tuning.
