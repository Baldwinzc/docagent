# Document Chunking

Documents are split into **chunks** before embedding because embeddings summarize a
bounded span and retrieval returns whole chunks. Strategies range from fixed-size
windows to structure-aware splits (by heading, paragraph, or sentence). Too large
dilutes the embedding and wastes context; too small loses surrounding meaning.
