# Mixture of Experts (MoE)

A **mixture-of-experts** layer replaces one FFN with many expert FFNs plus a
router that sends each token to a few experts (e.g. top-2). Only the chosen
experts run, so the model has a huge parameter count but a much smaller *active*
compute per token — decoupling capacity from cost. Load balancing the router is
the main training challenge.
