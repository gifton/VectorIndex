# Chapter 1: Similarity Search Fundamentals

> **Before you can search fast, you need to understand what you're searching for.**

This chapter establishes the mathematical and conceptual foundations for everything that follows. We'll explore what "similarity" means for vectors, why naive search doesn't scale, and how approximate methods bridge the gap.

---

## What You'll Learn

| Guide | Time | What You'll Learn |
|-------|------|-------------------|
| [1. Distance Metrics](./01-Distance-Metrics.md) | 12 min | L2, cosine, dot product—when to use each |
| [2. The Curse of Dimensionality](./02-Curse-Of-Dimensionality.md) | 10 min | Why high-dimensional search is fundamentally hard |
| [3. Approximate Nearest Neighbors](./03-Approximate-Nearest-Neighbors.md) | 8 min | The tradeoff that makes scale possible |

---

## The Big Picture

Consider a semantic search application:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           User Query                                    │
│                                                                         │
│   "What are the symptoms of vitamin D deficiency?"                      │
│                                                                         │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        Embedding Model                                  │
│                                                                         │
│   Query → [0.12, -0.45, 0.78, ..., 0.33]  (768 dimensions)             │
│                                                                         │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        Vector Index                                     │
│                                                                         │
│   Find k vectors most similar to the query                              │
│   from a database of 10,000,000 document embeddings                     │
│                                                                         │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        Search Results                                   │
│                                                                         │
│   1. "Vitamin D deficiency can cause fatigue, bone pain..."            │
│   2. "Common symptoms include muscle weakness..."                       │
│   3. "Low vitamin D levels are associated with..."                     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

The vector index is the critical component. It must:
1. **Understand similarity**: What makes two vectors "close"?
2. **Scale**: Handle millions of vectors
3. **Be fast**: Return results in milliseconds

---

## Why This Matters

### The Scale Problem

Modern embedding models produce vectors with hundreds of dimensions:
- **OpenAI text-embedding-3-small**: 1536 dimensions
- **Sentence-BERT**: 384-768 dimensions
- **CLIP image embeddings**: 512-768 dimensions

A typical application might have:
- 10 million documents
- 768-dimensional embeddings
- 4 bytes per float

That's **30 GB** of vector data. Searching it exhaustively for every query is expensive.

### The Accuracy Problem

Different applications need different notions of "similar":
- **Semantic search**: Meaning similarity (cosine)
- **Image retrieval**: Visual similarity (L2)
- **Recommendation**: Preference alignment (dot product)

Choosing the wrong metric silently degrades quality.

---

## 🔗 VectorCore Connection

This chapter builds directly on VectorCore concepts:

| VectorCore Concept | Application Here |
|-------------------|------------------|
| [SIMD operations](../../VectorCore/Guides/02-SIMD-Demystified/README.md) | Distance computations use SIMD4 |
| [Numerical stability](../../VectorCore/Guides/03-Numerical-Computing/README.md) | Avoiding overflow in L2 distance |
| [Cache-friendly access](../../VectorCore/Guides/01-Memory-Fundamentals/README.md) | Batch distance computation |

VectorCore taught you to compute one distance fast. This chapter teaches you which distance to compute and why you can't compute all of them.

---

## Start Here

**[→ Distance Metrics](./01-Distance-Metrics.md)**

---

*Chapter 1 of 7 • [Next: Flat Index Baseline →](../02-Flat-Index-Baseline/README.md)*
