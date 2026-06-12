# Byte-Level Tokenization

**Byte-level BPE** (GPT-2 onward) runs BPE over raw bytes rather than Unicode
characters, so the base vocabulary is just 256 bytes and *any* string — emoji,
code, other scripts — is representable with no unknown token.
