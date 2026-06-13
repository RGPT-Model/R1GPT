# Attention Mechanism in R1GPT

The core engine of R1GPT is the Self-Attention mechanism. It allows tokens in a sequence to dynamically route information to one another based on their content and position.

## 1. Mathematical Formulation

For an input sequence represented as a tensor of shape `(B, T, C)`, where:
- `B`: Batch size
- `T`: Sequence length (context/block size)
- `C`: Embedding dimension (`n_embd`)

We project the input `x` into three separate spaces using linear layers:
- **Query ($Q$)**: What this token is looking for.
- **Key ($K$)**: What information this token contains.
- **Value ($V$)**: The actual content to be routed if selected.

$$ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V $$

Where $d_k$ is the dimension of the head (`head_size`).

## 2. Why the Scaled Division?

The division by $\sqrt{d_k}$ (scaling factor) is crucial. As the head dimension grows, the dot product values can grow very large in magnitude. Large values feed into the `softmax` function, pushing it into regions with extremely small gradients (gradient vanishing). Dividing by $\sqrt{d_k}$ keeps the variance of the dot products around 1, preserving gradient flow.

## 3. Causal Masking (Decoder-only)

Since R1GPT is an autoregressive decoder model, tokens should only be allowed to attend to past and current tokens, never future tokens.

We enforce this by applying a lower-triangular mask (`tril`):
```python
# Create a lower triangular mask of size (T, T)
tril = torch.tril(torch.ones(T, T))
# Fill upper triangle (future tokens) with negative infinity
wei = wei.masked_fill(tril == 0, float('-inf'))
```

When `softmax` is applied, $e^{-\infty}$ becomes $0$, ensuring zero attention weights are allocated to future tokens.

## 4. Multi-Head Attention

Instead of running a single attention mechanism over the entire embedding dimension `n_embd`, we divide it into multiple heads (e.g. 4 heads, each with `head_size = n_embd / 4`).

Each head operates independently, allowing the model to attend to different parts of the context simultaneously (e.g. one head focusing on subject-verb agreement, another on punctuation, etc.). The outputs of all heads are concatenated together and projected back via a final projection layer:

```python
out = torch.cat([h(x) for h in self.heads], dim=-1)
out = self.proj(out)
```
