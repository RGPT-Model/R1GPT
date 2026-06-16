# Attention Mechanism in R1GPT

> The core engine of every modern language model. Self-attention allows each token in a sequence to dynamically gather information from every other token — forming context-aware representations that power coherent text generation.

---

## 1. The Core Intuition

Before attention, models processed each token in isolation. Attention solves the fundamental problem: **context**.

```
Without Attention:
  "bank" → fixed representation
   ↳ cannot tell: river bank? bank account?

With Attention:
  "river bank" → "bank" attends to "river"
   ↳ representation shifts toward geography meaning

  "bank account" → "bank" attends to "account"
   ↳ representation shifts toward finance meaning
```

Attention lets each token **ask the sequence a question** and gather a weighted blend of answers.

---

## 2. Query, Key, Value — The Library Analogy

The best mental model for Q, K, V:

```
┌─────────────────────────────────────────────────────┐
│                     Library                          │
│                                                      │
│  Query  =  "I want a book about physics"             │
│            (what this token is searching for)        │
│                                                      │
│  Key    =  Book spine labels / catalogue entries     │
│            (what each token advertises it contains)  │
│                                                      │
│  Value  =  Actual book contents                      │
│            (information passed if selected)          │
└─────────────────────────────────────────────────────┘
```

Each token simultaneously:
- **Broadcasts** a Key (what it contains)
- **Broadcasts** a Value (what it will share)
- **Emits** a Query (what it wants to find)

Attention scores match each Query against all Keys to decide how much of each Value to collect.

---

## 3. Mathematical Formulation

For an input tensor of shape `(B, T, C)`:
- `B` — Batch size
- `T` — Sequence length (`block_size = 64` in R1GPT v2)
- `C` — Embedding dimension (`n_embd = 128`)

**Step 1 — Project to Q, K, V:**

```python
self.key   = nn.Linear(n_embd, head_size, bias=False)
self.query = nn.Linear(n_embd, head_size, bias=False)
self.value = nn.Linear(n_embd, head_size, bias=False)

k = self.key(x)    # (B, T, head_size)
q = self.query(x)  # (B, T, head_size)
v = self.value(x)  # (B, T, head_size)
```

**Step 2 — Compute attention scores:**

```python
wei = q @ k.transpose(-2, -1)   # (B, T, T)
```

**Step 3 — Scale:**

```python
wei = wei * (head_size ** -0.5)
```

**Step 4 — Apply causal mask:**

```python
tril = torch.tril(torch.ones(T, T))
wei  = wei.masked_fill(tril == 0, float('-inf'))
```

**Step 5 — Softmax to probabilities:**

```python
wei = F.softmax(wei, dim=-1)     # (B, T, T)
```

**Step 6 — Weighted sum of values:**

```python
out = wei @ v                    # (B, T, head_size)
```

**The full formula:**

```
Attention(Q, K, V) = softmax( QKᵀ / √dₖ ) × V
```

---

## 4. Why Scale by √dₖ?

Without scaling, dot products grow proportionally to the head dimension:

```
head_size = 32:
  Typical Q·K ≈ 32 × (unit variance) = variance of 32
  std ≈ √32 ≈ 5.7

Without scaling → large values → softmax saturates:
  softmax([20, 3, 1]) ≈ [1.0, 0.0, 0.0]  ← one-hot, no gradient
  softmax([ 2, 0.3, 0.1]) ≈ [0.8, 0.1, 0.1]  ← smooth, good gradient

Fix: divide by √dₖ = √32 ≈ 5.66
  Keeps dot product variance ≈ 1 → softmax stays in useful range
```

| Condition | Softmax output | Gradient flow |
|---|---|---|
| Large unscaled scores | Near one-hot | ❌ Vanishes |
| Scaled scores (÷√dₖ) | Smooth distribution | ✅ Healthy |

---

## 5. Causal Masking — No Looking Ahead

R1GPT is a **decoder-only** autoregressive model. During training, predicting token at position `t` must not use information from positions `t+1, t+2, ...`

We enforce this with a lower-triangular mask:

```
T=5 Causal Mask:
         attend to:
         pos0  pos1  pos2  pos3  pos4
  pos0 [  ✓    -∞    -∞    -∞    -∞  ]
  pos1 [  ✓     ✓    -∞    -∞    -∞  ]
  pos2 [  ✓     ✓     ✓    -∞    -∞  ]
  pos3 [  ✓     ✓     ✓     ✓    -∞  ]
  pos4 [  ✓     ✓     ✓     ✓     ✓  ]
```

After `softmax(-∞) = 0`, future positions receive **exactly zero** attention weight — they are invisible to the model.

```python
tril = torch.tril(torch.ones(T, T))
wei  = wei.masked_fill(tril == 0, float('-inf'))
wei  = F.softmax(wei, dim=-1)
# Upper triangle is now exactly 0.0
```

---

## 6. Softmax — From Scores to Probabilities

Raw attention scores have no probabilistic meaning. Softmax converts them:

```
Before softmax:  [-2.3,  1.4,  0.8,  -inf]
After softmax:   [ 0.05, 0.70, 0.25,   0.0]
                                          ↑ future token, zeroed out
```

**Properties guaranteed:**
- All values ∈ [0, 1]
- All values sum to exactly 1.0
- High scores dominate, low scores fade, `-inf` → 0

---

## 7. Multi-Head Attention

A single attention head can only capture one type of relationship at a time. R1GPT uses **4 parallel heads**, each with `head_size = 32`:

```
Input (B, T, 128)
        │
  ┌─────┴──────────────────────────────────┐
  │          Split along embedding dim     │
  └──┬──────────┬──────────┬──────────┬───┘
     │          │          │          │
  Head 1     Head 2     Head 3     Head 4
  (B,T,32)  (B,T,32)  (B,T,32)  (B,T,32)
     │  ↓       │  ↓       │  ↓       │  ↓
     Attn      Attn      Attn      Attn
     │          │          │          │
  └──┴──────────┴──────────┴──────────┘
              Concatenate
              (B, T, 128)
                  │
           Linear Projection
                  │
           Output (B, T, 128)
```

Different heads learn to specialize:

```
Head 1 → Subject-verb agreement, syntax
Head 2 → Long-range coreference (pronouns → nouns)
Head 3 → Local phrase structure
Head 4 → Punctuation and sentence boundary patterns
```

```python
out = torch.cat([h(x) for h in self.heads], dim=-1)
out = self.proj(out)   # linear projection back to n_embd
```

---

## 8. Dropout in Attention (v2)

After the projection layer, dropout is applied to prevent the model from relying on any single attention pattern:

```python
self.dropout = nn.Dropout(dropout)   # p=0.2

# Applied after softmax weights and after projection:
wei = self.dropout(wei)
out = self.dropout(self.proj(out))
```

During training, 20% of attention weights are randomly zeroed — forcing the model to distribute information across multiple heads and pathways.

---

## 9. Attention Complexity

| Aspect | Complexity | Notes |
|---|---|---|
| Time | O(T² × C) | Quadratic in sequence length |
| Memory | O(T²) | The (B, T, T) weight matrix |
| `block_size=64` | 64² = 4,096 pairs | Manageable on CPU |
| GPT-4 equivalent | ~100K² pairs | Requires FlashAttention |

> The quadratic cost is why context length is hard to scale. Research into linear attention, sparse attention, and FlashAttention all aim to reduce this bottleneck.

---

## 10. Complete Attention Head — R1GPT Implementation

```python
class Head(nn.Module):

    def __init__(self, head_size):
        super().__init__()
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer(
            'tril',
            torch.tril(torch.ones(block_size, block_size))
        )

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)     # (B, T, head_size)
        q = self.query(x)   # (B, T, head_size)
        v = self.value(x)   # (B, T, head_size)

        # Scaled dot-product attention
        wei = q @ k.transpose(-2, -1) * (C ** -0.5)  # (B, T, T)

        # Causal mask
        wei = wei.masked_fill(
            self.tril[:T, :T] == 0,
            float('-inf')
        )
        wei = F.softmax(wei, dim=-1)  # (B, T, T)

        # Weighted aggregation
        out = wei @ v   # (B, T, head_size)
        return out
```

---

## Summary

```
Input x (B, T, C)
   │
   ├─→ Linear → Q (B, T, d_k)
   ├─→ Linear → K (B, T, d_k)
   └─→ Linear → V (B, T, d_k)
                │
           Q @ Kᵀ → raw scores (B, T, T)
                │
           ÷ √d_k → scaled scores
                │
         + causal mask (-inf future)
                │
           softmax → weights (B, T, T)  ← each row sums to 1
                │
        weights @ V → output (B, T, d_k)
```

> Attention is the mechanism by which every token gets to ask: *"What does the rest of the sequence know that I need?"* — and gets a weighted, context-sensitive answer.
