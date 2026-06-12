# Recall@k

**Recall@k** measures whether the passages needed to answer appear among the top-k
retrieved. It is the ceiling on answer quality: evidence never retrieved cannot be
used. For multi-hop questions recall must hold for every needed passage, so it is
typically lowest there.
