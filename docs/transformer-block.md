# Transformer Block in R1GPT

> The Transformer Block is the fundamental repeating unit of the GPT architecture. It combines multi-head self-attention, a position-wise feed-forward network, layer normalization, residual connections, and dropout into a single composable module. R1GPT v2 stacks 4 of these blocks.

---

## 1. Full Block Structure

```
Input x  (B, T, 128)
   │
   │   ┌─────────────────────────────────────────────┐
   ├──►│              Attention Sublayer              │
   │   │                                             │
   │   │  LayerNorm(x)                               │
   │   │      ↓                                      │
   │   │  MultiHeadAttention                         │
   │   │  (4 heads, head_size=32, dropout=0.2)       │
   │   │      ↓                                      │
   │   │  + x  (residual connection)                 │
   │   └─────────────────────────────────────────────┘
   │                       │
   │   ┌─────────────────────────────────────────────┐
   └──►│            Feed-Forward Sublayer             │
       │                                             │
       │  LayerNorm(x)                               │
       │      ↓                                      │
       │  Linear(128 → 512)                          │
       │      ↓                                      │
       │  ReLU                                       │
       │      ↓                                      │
       │  Linear(512 → 128)                          │
       │      ↓                                      │
       │  Dropout(0.2)                               │
       │      ↓                                      │
       │  + x  (residual connection)                 │
       └─────────────────────────────────────────────┘
                           │
               Output x  (B, T, 128)
```

**In code:**

```python
def forward(self, x):
    x = x + self.sa(self.ln1(x))    # attention sublayer
    x = x + self.ffwd(self.ln2(x))  # feed-forward sublayer
    return x
```

---

## 2. Pre-LN vs. Post-LN

The original "Attention Is All You Need" paper used **Post-LayerNorm**:

```
Post-LN (original 2017 paper):
  x → SubLayer(x) → + x → LayerNorm → output
```

Modern architectures (GPT-2, GPT-3, R1GPT) use **Pre-LayerNorm**:

```
Pre-LN (R1GPT v2):
  x → LayerNorm(x) → SubLayer → + x → output
```

### Why Pre-LN Wins

```
Post-LN residual path:
  gradient → LayerNorm ← bottleneck (normalized, scaled, shifted)
  
Pre-LN residual path:
  gradient → direct ← clean highway, no bottleneck
```

| Property | Post-LN | Pre-LN |
|---|---|---|
| Residual path | Goes through LayerNorm | **Direct, clean** |
| Gradient flow | Can vanish in deep models | **Stable in deep models** |
| Training stability | Needs careful LR warmup | **More robust** |
| Used by | Original Transformer (2017) | GPT-2, GPT-3, R1GPT |

**Formula comparison:**

```
Post-LN:  x_out = LayerNorm(x + SubLayer(x))
Pre-LN:   x_out = x + SubLayer(LayerNorm(x))    ← R1GPT
```

In Pre-LN, the residual addition `+ x` always operates on un-normalized activations. Gradients can flow directly from the loss back to the earliest embedding layer without passing through any LayerNorm operations. This is essential for training 4+ layer networks stably.

---

## 3. Residual Connections — The Gradient Highway

Residual (skip) connections are one of the most impactful architectural ideas in deep learning:

```
Without residual:
  Layer 1 → Layer 2 → Layer 3 → Layer 4 → Loss
  Gradient must pass through all 4 layers to reach Layer 1
  → Vanishing gradient risk

With residual:
  x → [Layer] → x + Δx → [Layer] → x + Δx → ... → Loss
       └──────────────────────────────────────────┘
                    Direct gradient highway
  Gradient can skip directly to any earlier layer
  → Stable training regardless of depth
```

```python
# The entire attention sublayer boils down to:
x = x + self.sa(self.ln1(x))

# self.sa() only needs to learn the DELTA (improvement),
# not reconstruct the entire representation from scratch.
# Initially, when weights are random, self.sa() ≈ 0 → x ≈ x
# The network starts as an identity function and gradually learns.
```

**Key insight:** Early in training, residual blocks effectively act as identity functions. The model learns incremental refinements at each layer rather than complete transformations.

---

## 4. Feed-Forward Network (FFWD)

After attention has aggregated information *across* tokens, the feed-forward network processes each token *independently*:

```
Attention:     "Communicate"  — tokens talk to each other
Feed-Forward:  "Think"        — each token processes what it heard
```

### Architecture (R1GPT v2 with Dropout)

```
Token representation (128-dim)
          │
   Linear(128 → 512)        ← expand 4× for representational capacity
          │
        ReLU()               ← introduce non-linearity
          │
   Linear(512 → 128)        ← compress back to model dimension
          │
   Dropout(p=0.2)            ← regularization (v2 addition)
          │
   + Residual connection
          │
  Output (128-dim)
```

```python
self.net = nn.Sequential(
    nn.Linear(n_embd, 4 * n_embd),    # 128 → 512
    nn.ReLU(),
    nn.Linear(4 * n_embd, n_embd),    # 512 → 128
    nn.Dropout(dropout),               # p=0.2
)
```

### Why 4× Expansion?

The expansion factor of 4 is a design choice from the original Transformer paper, validated empirically across many architectures:

| Expansion | Hidden Size | Parameters (per FFWD) | Capacity |
|---|---|---|---|
| 1× | 128 | ~33K | Low |
| 2× | 256 | ~66K | Moderate |
| **4×** | **512** | **~132K** | **Standard (GPT)** |
| 8× | 1024 | ~264K | High (GPT-3 style) |

The expanded hidden dimension gives the network room to learn complex non-linear combinations of the features gathered during attention. Without this expansion, the FFWD would be nearly equivalent to a single linear transformation.

### ReLU vs GELU

R1GPT v2 uses ReLU for simplicity. GPT-2 and GPT-3 use GELU:

```
ReLU:  f(x) = max(0, x)          — hard zero for x < 0
GELU:  f(x) = x · Φ(x)           — smooth, probabilistic gating
```

GELU generally performs slightly better on language tasks. A future upgrade could swap `nn.ReLU()` for `nn.GELU()`.

---

## 5. Layer Normalization

Activations in deep networks can have wildly different scales across features. LayerNorm stabilizes this:

```
Before LayerNorm:
  feature values: [0.002,  4500,  -1.2,  890,  0.05]
                   ← completely different magnitudes ←

After LayerNorm:
  feature values: [-0.8,   1.1,  -0.7,   0.9, -0.5]
                   ← normalized to roughly unit scale ←
```

**Formula:**

```
LayerNorm(x) = γ · (x - μ) / (σ + ε) + β

Where:
  μ = mean of x across features
  σ = std of x across features
  γ, β = learnable scale and shift parameters
  ε = 1e-5 (numerical stability)
```

Unlike BatchNorm (which normalizes across the batch dimension), LayerNorm normalizes **within each token's feature vector** — making it independent of batch size and sequence length. This is ideal for language models where batch sizes can vary.

---

## 6. Dropout — Regularization (v2)

Dropout was added in v2 to combat overfitting on larger datasets:

```
Training (dropout active):
  [0.3,  0.0,  0.8,  0.0,  1.2,  0.0,  0.5]
        ↑ zeroed      ↑ zeroed      ↑ zeroed
  20% of neurons randomly silenced each forward pass

Inference (dropout disabled):
  [0.3,  0.4,  0.8,  0.6,  1.2,  0.7,  0.5]
  All neurons active, weights unchanged
```

Dropout is applied in two places:
1. **After the attention projection** — `MultiHeadAttention.dropout`
2. **After the FFWD second linear** — `FeedForward.net[-1]` (Dropout layer)

**Effect on R1GPT v2 training:**

```
Train loss ≈ Val loss throughout 5000 steps
→ Dropout successfully suppresses overfitting
```

---

## 7. Stacking 4 Blocks

```
Embedding
   │
   ▼
Block 1 → learns basic character/token patterns
   │
   ▼
Block 2 → learns word-level combinations
   │
   ▼
Block 3 → learns phrase and clause structure
   │
   ▼
Block 4 → learns longer-range dependencies
   │
   ▼
Final LayerNorm
   │
   ▼
LM Head → logits (vocab_size)
```

Each block sees the output of the previous block — progressively building from low-level to high-level representations. This hierarchical composition is what gives Transformers their expressive power.

---

## 8. Complete Block — R1GPT v2 Implementation

```python
class Block(nn.Module):

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa   = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1  = nn.LayerNorm(n_embd)
        self.ln2  = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))    # attention + residual
        x = x + self.ffwd(self.ln2(x))  # ffwd + residual
        return x


class FeedForward(nn.Module):

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)
```

---

## Summary

| Component | Role | Why it matters |
|---|---|---|
| **Pre-LayerNorm** | Normalize before sublayer | Clean residual path, stable gradients |
| **Multi-Head Attention** | Token communication | Context-aware representations |
| **Residual Connection** | Skip connection | Gradient highway, identity initialization |
| **Feed-Forward (4×)** | Token-wise processing | Non-linear feature combination |
| **Dropout (0.2)** | Regularization | Prevents overfitting on large data |
| **Stacking ×4** | Depth | Hierarchical representation learning |
