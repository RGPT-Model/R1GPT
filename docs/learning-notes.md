# R1GPT Learning Notes

> This document captures the major concepts, implementation details, debugging lessons, and architectural insights gained while building **R1GPT** — a GPT-style Transformer language model implemented completely from scratch in PyTorch.

---

## 🎯 Project Goal

The purpose of R1GPT was not merely to reproduce NanoGPT code, but to deeply understand:

| Goal | Description |
|------|-------------|
| 🧠 Language Models | How they work internally |
| ⚙️ Transformers | How they process text |
| 🔍 Attention | How it works mathematically |
| 📈 Training | How gradient updates teach the model |
| 📝 Generation | How text emerges from next-token prediction |

---

## Language Modeling Is Next Token Prediction

The most important realization from the entire project:

> **GPT is not trained to "understand" language directly. GPT is trained to predict the next token.**

```
Current Tokens  →  Next Token
─────────────────────────────
"I love"        →  "AI"
"The capital of France is"  →  "Paris"
```

During training, billions of these prediction tasks teach the model statistical patterns of language.

---

## Text Must Become Numbers

Neural networks cannot process raw text. The pipeline is:

```
Text
 ↓
Tokenizer
 ↓
Token IDs (integers)
```

**Example — character-level tokenization:**

```
"banana"
 ↓
Vocabulary: { 'a':0, 'b':1, 'n':2 }
 ↓
[1, 0, 2, 0, 2, 0]
```

This was the first step toward transforming language into something a neural network can learn from.

---

## Vocabulary Construction

The vocabulary is built in one line:

```python
chars = sorted(list(set(text)))
```

Each step serves a specific purpose:

| Step | Operation | Why |
|------|-----------|-----|
| `set()` | Remove duplicates | Each character appears once |
| `list()` | Enable indexing | Allows enumeration |
| `sorted()` | Deterministic order | Same vocab every run |

> Without converting to a list, indexing and enumeration would not be possible.

---

## STOI and ITOS — Bidirectional Mapping

Two dictionaries bridge text and numbers:

```python
stoi  # String → Integer
itos  # Integer → String
```

**Example:**

```python
stoi = { 'a':0, 'b':1, 'c':2 }
itos = { 0:'a', 1:'b', 2:'c' }
```

This creates a complete round-trip:

```
Text → encode() → Model → decode() → Text
```

---

## Dataset Creation

The model does not learn from entire texts at once. Instead, every sequence is sliced into input/target pairs:

```python
x = data[i : i + block_size]      # input
y = data[i+1 : i + block_size + 1]  # target (shifted by 1)
```

**Concrete example:**

```
x = [ h, e, l, l ]
y = [ e, l, l, o ]
```

Each position predicts the next token, creating thousands of learning examples from a single corpus.

---

## Block Size — The Context Window

```python
block_size = 8
```

This means: **the model can look at the previous 8 tokens.**

```
Tokens: [ a, b, c, d, e, f, g, h ]
                                 ↓
                          predict 'i'
```

Increasing block size gives richer context but requires more memory and compute:

| `block_size` | Context |
|---|---|
| `8` | 8 tokens back |
| `64` | 64 tokens back |
| `2048` | 2048 tokens back (GPT-3 scale) |

> Context length is one of the major factors affecting model quality.

---

## Batching

Instead of one sequence per update:

```
1 sequence → 1 gradient update  ❌ (slow, noisy)
```

We process many in parallel:

```python
batch_size = 4
```

```
(B, T) = (4, 8)
↓
4 sequences × 8 tokens each
processed simultaneously
```

**Benefits:**
- ⚡ Faster GPU utilization
- 📊 More stable gradients
- 🔄 Better training efficiency

---

## Embeddings

Raw token IDs carry no semantic meaning:

```
cat = 15
dog = 42
```

These numbers tell the model nothing about similarity. Embeddings fix this:

```
Token ID
  ↓
Embedding Layer  (learnable lookup table)
  ↓
Dense Float Vector

"cat" → [0.12, -0.44, 1.02, 0.77, ...]
```

Over training, vectors for similar concepts cluster closer together in embedding space.

---

## Why Bigram Models Fail

The first model built was a simple Bigram:

```
Current Token → Predict Next Token
```

**Problem:** No memory of prior context.

```
"bank" could mean:
  ├─ "river bank"    (geography)
  └─ "bank account"  (finance)
```

Without context, the model cannot distinguish these. This limitation directly motivates **Attention**.

---

## Attention — Solving the Memory Problem

Attention allows tokens to **communicate** with each other:

```
Before Attention:
  Each token processes only itself.

After Attention:
  Each token can attend to all previous tokens
  and gather relevant information.
```

```
Input Tokens
  ↓
Attention Mechanism
  ↓
Context-Aware Representation
```

This is the foundation of modern LLMs.

---

## Query, Key, Value — The Mental Model

The most useful analogy for understanding attention:

```
┌─────────────────────────────────────────────┐
│                  Library                    │
│                                             │
│  Query  =  "I want a book about physics"    │
│  Key    =  Book spine labels / topics       │
│  Value  =  Actual book contents             │
└─────────────────────────────────────────────┘
```

Attention computes how strongly each **Query** matches each **Key**, and uses those scores to blend the **Values**.

```python
self.key   = nn.Linear(n_embd, head_size, bias=False)
self.query = nn.Linear(n_embd, head_size, bias=False)
self.value = nn.Linear(n_embd, head_size, bias=False)
```

---

## Attention Matrix

**Shapes involved:**

```
Q : (B, T, d_k)
K : (B, T, d_k)
V : (B, T, d_k)
```

**Dot-product attention:**

```python
wei = q @ k.transpose(-2, -1)  # (B, T, T)
wei = wei * (d_k ** -0.5)      # scale
wei = softmax(wei)              # probabilities
out = wei @ v                  # (B, T, d_k)
```

The `(B, T, T)` weight matrix means: **every token compares itself with every other token.**

> ⚠️ Scaling by `1/√d_k` prevents dot products from growing too large, which would push softmax into vanishing-gradient territory.

---

## Causal Masking — No Cheating

GPT is a **decoder-only** model. It must not look at future tokens:

```
Predicting position 3:
  ✅ Can see: tokens 0, 1, 2
  ❌ Must not see: tokens 4, 5, 6, 7
```

This is enforced with a lower-triangular mask:

```python
tril = torch.tril(torch.ones(T, T))
wei  = wei.masked_fill(tril == 0, float('-inf'))
```

After `softmax(-inf) = 0`, future positions receive zero attention weight. This ensures **autoregressive** learning.

```
Mask visualization (T=4):
  1  -∞  -∞  -∞
  1   1  -∞  -∞
  1   1   1  -∞
  1   1   1   1
```

---

## Softmax Creates Probabilities

Raw attention scores have no probabilistic meaning:

```
Before softmax:  [-2.3,  1.4,  0.8]
After softmax:   [ 0.05, 0.70, 0.25]
```

**Properties guaranteed by softmax:**
- All values ≥ 0
- All values sum to exactly 1
- High scores dominate, low scores fade

---

## Multi-Head Attention

A single attention head captures one type of relationship. Running **multiple heads in parallel** allows the model to attend to different aspects simultaneously:

```
Head 1 → Grammar & syntax
Head 2 → Long-range dependencies
Head 3 → Entity relationships
Head 4 → Local sentence structure
```

Implementation:

```python
out = torch.cat([h(x) for h in self.heads], dim=-1)  # concat
out = self.proj(out)  # project back to n_embd
```

> This is one of the core innovations that made Transformers dominant.

---

## Feed-Forward Networks

After attention, tokens have gathered information from each other. Now each token **thinks independently**:

```
Attention: "Communicate"
Feed-Forward: "Think"
```

```python
self.net = nn.Sequential(
    nn.Linear(n_embd, 4 * n_embd),   # expand
    nn.ReLU(),                         # non-linearity
    nn.Linear(4 * n_embd, n_embd)    # compress back
)
```

The **4× expansion** gives the model room to learn complex non-linear combinations of attended features.

---

## Residual Connections

One of the most important training tricks:

```python
x = x + layer(x)   # residual connection
```

Instead of discarding original information:

```
Without residual:  x → layer → new_x   (original lost)
With residual:     x → layer → x + new_x  (original preserved)
```

**Why it matters:**
- 🌊 Gradients flow directly from output to input
- 🏗️ Enables training very deep networks
- 🎯 Easier optimization landscape

---

## Layer Normalization

Activations across a layer can have wildly different scales. LayerNorm stabilizes this:

```
Raw activations: [0.002, 4500, -1.2, 890]   ← unstable
After LayerNorm: [-0.3,   1.1, -0.8,  0.9]  ← stable
```

**Pre-LayerNorm (used in R1GPT):**

```
LayerNorm → Attention → + Residual
LayerNorm → FeedForward → + Residual
```

Applying normalization *before* the sublayer keeps the residual path clean, which improves gradient flow in deep architectures.

---

## Transformer Block — The Full Picture

```
Input x
  │
  ├─→ LayerNorm(x) → MultiHeadAttention ─→ + x  (residual)
  │                                          │
  └─→ LayerNorm(x) → FeedForward        ─→ + x  (residual)
                                             │
                                           Output
```

```python
def forward(self, x):
    x = x + self.sa(self.ln1(x))    # attention sublayer
    x = x + self.ffwd(self.ln2(x))  # ffwd sublayer
    return x
```

Stacking multiple blocks creates increasingly powerful representations.

---

## Positional Embeddings

Transformers process all tokens **in parallel**, which means position information is lost by default:

```
"good not" vs "not good"  →  identical to a pure Transformer
```

Fix: add a learnable positional embedding to each token:

```python
x = tok_emb + pos_emb
```

| Embedding | Encodes |
|-----------|---------|
| `tok_emb` | What token? |
| `pos_emb` | Where in the sequence? |

The model learns both simultaneously.

---

## Training Loop

```
┌─────────────────────────────────────┐
│           Training Loop             │
│                                     │
│  get_batch()                        │
│       ↓                             │
│  forward(x, y)  → logits, loss      │
│       ↓                             │
│  optimizer.zero_grad()              │
│       ↓                             │
│  loss.backward()  → gradients       │
│       ↓                             │
│  optimizer.step() → update weights  │
│       ↓                             │
│  Repeat ×1000                       │
└─────────────────────────────────────┘
```

**Optimizer used:** `AdamW` — the standard for modern Transformer training, combining adaptive learning rates with weight decay.

---

## Text Generation

Training and generation are **fundamentally different processes**:

```
Training:
  Input + Target → Compare → Learn (backwards pass)

Generation:
  Input → Predict → Append → Repeat (no backwards pass)
```

This loop is called **Autoregressive Generation** — the "G" in GPT stands for *Generative*, referring to exactly this process.

```python
def generate(self, idx, max_new_tokens):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]       # trim context
        logits, _ = self(idx_cond)
        logits = logits[:, -1, :]             # last token only
        probs = F.softmax(logits, dim=-1)     # distribution
        idx_next = torch.multinomial(probs, 1) # sample
        idx = torch.cat([idx, idx_next], dim=1)
    return idx
```

---

## The Most Important Lesson

Building a GPT from scratch revealed that **large language models are not magic**.

They are composed of understandable components:

```
"hello world\ni love ai\ngpt is amazing"
  ↓
Character Tokenization       (text → integers)
  ↓
Embeddings                   (integers → vectors)
  ↓
Positional Embeddings        (add position signal)
  ↓
Multi-Head Attention ×3      (tokens communicate)
  ↓
Feed-Forward Networks ×3     (tokens process)
  ↓
LayerNorm + Residuals        (stabilize training)
  ↓
LM Head                      (vectors → logits)
  ↓
Cross-Entropy Loss            (compare with target)
  ↓
AdamW Gradient Update        (learn)
  ↓
Autoregressive Generation    (produce text)
```

Every component serves a specific, understandable purpose. Understanding them individually makes modern AI systems significantly less mysterious.

---

## Final Outcome — R1GPT v1

| Component | Status |
|-----------|--------|
| Character Tokenization | ✅ |
| Vocabulary Construction | ✅ |
| Dataset Batching | ✅ |
| Self-Attention Head | ✅ |
| Multi-Head Attention | ✅ |
| Feed-Forward Network | ✅ |
| Residual Connections | ✅ |
| Layer Normalization | ✅ |
| Transformer Block | ✅ |
| Positional Embeddings | ✅ |
| Cross-Entropy Loss | ✅ |
| AdamW Optimizer | ✅ |
| Autoregressive Generation | ✅ |

> **Result: A fully functional GPT-style Transformer — implemented from scratch in PyTorch.**
