# R1GPT Learning Notes

This document captures key insights, implementation pitfalls, and takeaways gained from building the R1GPT Transformer architecture from scratch.

## 1. Sequence Order & Positional Embeddings

Since Transformers process all tokens in parallel, they have no inherent sense of sequence order. Without positional embeddings, the sentence `"not good, but bad"` and `"good, but not bad"` would yield identical representations.

We solve this by adding a learnable **Positional Embedding** ($P$) to the **Token Embedding** ($T$):
```python
x = tok_emb + pos_emb
```
Where:
- `tok_emb` has shape `(B, T, C)`
- `pos_emb` has shape `(T, C)` (broadcasting across batch size `B`)

## 2. Shape Integrity & Transposition Gotchas

Tensor dimensions must be tracked carefully:
1. When computing attention weights:
   $$ Q(B, T, d_k) \times K^T(B, d_k, T) \rightarrow \text{weights}(B, T, T) $$
   In code:
   `wei = q @ k.transpose(-2, -1)`
   *Mistake to avoid*: transposing dims `(0, 1)` instead of the last two dimensions `(-2, -1)`.
2. When applying cross-entropy loss:
   PyTorch expects the logits to be of shape `(N, C)` where $N$ is batch size and $C$ is number of classes. We flatten our `(B, T, C)` logits to `(B*T, C)` and `(B, T)` targets to `(B*T)` for loss evaluation.

## 3. The Power of Overfitting as a Sanity Check

One of the most effective ways to debug deep learning architectures is to train the model on a tiny dataset (like a single paragraph or 3 lines of text) and ensure that it can overfit to a loss close to zero.

If the loss does not fall:
- There is a bug in the gradient computation (e.g. detached tensors).
- The causal mask is incorrectly applied, causing information leakage or wrong shape logic.
- Residual connections are not summing correctly.

In our case, testing on the tiny 3-line corpus of `input.txt` allowed the model to rapidly overfit (loss $\approx 0.07$) and reconstruct the exact sentences perfectly. This proved the mathematical and structural correctness of the architecture.
