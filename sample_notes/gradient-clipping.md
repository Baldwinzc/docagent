# Gradient Clipping

**Gradient clipping** rescales gradients whose global norm exceeds a threshold,
capping the size of any single update. It prevents rare exploding gradients from
derailing training and is a cheap, standard safeguard for deep sequence models.
