# IVF Index

**IVF** (inverted file) clusters vectors with k-means and, at query time, searches
only the few clusters nearest the query. Probing more clusters raises recall but
costs more time — a direct speed/recall knob.
