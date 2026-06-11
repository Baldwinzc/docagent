# Evaluating Retrieval-Augmented Generation Systems

## Why evaluation is hard

A RAG system has two stages that can each fail, so a single end-to-end score
hides where the problem is. Evaluation therefore separates **retrieval quality**
(did we fetch the right passages?) from **answer quality** (given those passages,
did we answer well and cite correctly?). A good answer built on lucky retrieval,
or a bad answer despite good retrieval, are different bugs.

## Retrieval metrics

- **Recall@k** measures whether the passages needed to answer appear in the top-k
  retrieved chunks. It is the ceiling on answer quality: if the evidence was never
  retrieved, the generator cannot ground an answer on it.
- **Precision@k** measures how many of the top-k chunks are actually relevant;
  low precision wastes the generator's context window on noise.
- For multi-hop questions, recall must hold for *every* passage the question
  needs, so recall is typically lowest on multi-hop cases.

## Answer metrics

- **Answer correctness** asks whether the final answer is right. Because answers
  are free text, this is often graded by an **LLM-as-judge**: a separate model
  scores the answer against a written criterion. It is cheap and scalable but can
  be inconsistent, so criteria should be specific.
- **Citation grounding** checks that the answer's citations point to sources that
  actually support it. A stricter version verifies each *sentence* is entailed by
  the chunk it cites, not just that the source was retrieved.
- **Hallucination rate** counts claims (or citations) the evidence does not
  support. Verifying emitted citations against what was retrieved, and dropping
  unsupported ones, drives this toward zero.

## Abstention and refusal

A system that always answers will confidently answer questions the corpus does
not cover. **Refusal accuracy** (or abstention) measures whether the system
correctly declines on out-of-scope or unanswerable questions instead of
fabricating. A relevance threshold on retrieval scores is one principled way to
abstain: if nothing clears the threshold, say so.

## Building an eval set

A useful labelled set spans categories: single-passage facts, multi-hop questions
that need several sources, out-of-scope questions (should be declined), and
in-domain-sounding but unanswerable questions (the corpus lacks the answer).
Reporting metrics **per category** matters because an aggregate can hide that, say,
multi-hop recall is poor while single-passage recall is perfect.
