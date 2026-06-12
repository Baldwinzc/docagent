# Direct Preference Optimization (DPO)

**DPO** aligns a model to preference pairs *without* training a separate reward
model or running RL: it derives a simple classification-style loss that directly
increases the likelihood of preferred responses over rejected ones. It is simpler
and more stable than PPO-based RLHF while reaching comparable quality.
