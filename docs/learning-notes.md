# R1GPT Learning Notes

> This document captures every major concept, implementation detail, debugging lesson, and architectural insight gained while building **R1GPT** — a GPT-style Transformer language model implemented completely from scratch in PyTorch, trained on WikiText with a custom BPE tokenizer.

---

## 🎯 Project Goal

The purpose of R1GPT was not merely to reproduce NanoGPT code, but to deeply understand every component of a modern language model:

| Goal | Description | Status |
|------|-------------|--------|
| 🧠 Language Models | How they work internally | ✅ |
| ⚙️ Transformers | How they process text | ✅ |
| 🔍 Attention | How it works mathematically | ✅ |
| 🔤 Tokenization | BPE from scratch | ✅ |
| 📈 Training | Gradient updates, scheduling | ✅ |
| 🛡️ Regularization | Dropout, validation loss | ✅ |
| 📝 Generation | Temperature, top-k sampling | ✅ |

---

## Language Modeling Is Next Token Prediction

The most important realization from the entire project:

> **GPT is not trained to "understand" language directly. GPT is trained to predict the next token.**

```
Current Tokens            →  Next Token
──────────────────────────────────────
"I love"                  →  "AI"
"The capital of France"   →  "is"
"The Eiffel Tower is in"  →  "Paris"
```

During training, billions of these prediction tasks teach the model statistical patterns of language — and understanding *emerges* from this process.

---

## Text Must Become Numbers

Neural networks cannot process raw text. The pipeline is:

```
Raw Text
   ↓
Tokenizer  (split into tokens)
   ↓
Token IDs  (map tokens → integers)
   ↓
Embeddings (map integers → vectors)
   ↓
Neural Network
```

**Two tokenization approaches used in R1GPT:**

| Approach | v1 | v2 |
|---|---|---|
| Method | Character-level | BPE (Byte-Pair Encoding) |
| Unit | Single character | Subword / merged token |
| Vocabulary | 17 chars (toy corpus) | 165 tokens (WikiText) |
| Granularity | Maximum | Learned from data |

---

## Character-Level Tokenization (v1)

The simplest approach — every character is its own token:

```python
chars = sorted(list(set(text)))
# ['a', 'b', 'c', 'd', ...]

stoi = { ch: i for i, ch in enumerate(chars) }
itos = { i: ch for i, ch in enumerate(chars) }

def encode(s): return [stoi[c] for c in s]
def decode(l): return ''.join([itos[i] for i in l])
```

**Example:**

```
"banana"
   ↓
Vocabulary: { 'a':0, 'b':1, 'n':2 }
   ↓
[1, 0, 2, 0, 2, 0]
```

**Limitation:** Every individual character gets its own ID, even common subwords like "the", "ing", "tion". The vocabulary is tiny but the sequences are long.

---

## BPE Tokenization (v2)

Byte-Pair Encoding learns to merge frequent character pairs into subword tokens, reducing sequence length while maintaining expressiveness.

### The Algorithm

```
Start:    ["h", "e", "l", "l", "o", " ", "w", "o", "r", "l", "d"]

Step 1:   Find most frequent pair: ("h", "e")
          Merge → "he"
          ["he", "l", "l", "o", " ", "w", "o", "r", "l", "d"]

Step 2:   Find most frequent pair: ("he", "l")
          Merge → "hel"
          ["hel", "l", "o", " ", "w", "o", "r", "l", "d"]

...

Step 4:   ["hello", " ", "w", "o", "r", "l", "d"]
```

### get_stats() — Count Pair Frequencies

```python
def get_stats(tokens):
    pairs = Counter()
    for i in range(len(tokens) - 1):
        pairs[(tokens[i], tokens[i+1])] += 1
    return pairs
```

### merge() — Apply a Merge Rule

```python
def merge(tokens, pair):
    new_tokens = []
    i = 0
    while i < len(tokens):
        if (i < len(tokens) - 1
                and tokens[i] == pair[0]
                and tokens[i+1] == pair[1]):
            new_tokens.append(pair[0] + pair[1])
            i += 2       # skip both consumed tokens
        else:
            new_tokens.append(tokens[i])
            i += 1
    return new_tokens
```

> ⚠️ Critical bug to avoid: using `i += 1` instead of `i += 2` when a merge happens causes double-counting and corrupts the token stream.

### What the Model Learned (First 20 WikiText Merges)

```
Merge  1:  (e, ' ')  → "e "    end-of-word: "the ", "are "
Merge  2:  (t, h)    → "th"    "the", "that", "this", "with"
Merge  3:  (t, ' ')  → "t "    end-of-word: "not ", "but "
Merge  4:  (s, ' ')  → "s "    plural/verb: "words ", "runs "
Merge  5:  (d, ' ')  → "d "    past tense: "called ", "used "
Merge  6:  (,, ' ')  → ", "    comma-space: universal pattern
Merge  7:  (o, u)    → "ou"    "out", "our", "you", "found"
Merge  8:  (e, r)    → "er"    "other", "over", "after"
Merge  9:  (i, n)    → "in"    "in", "into", "this"
Merge 10:  (y, ' ')  → "y "    "they ", "by ", "only "
Merge 18:  (' ', th) → " th"   " the", " this", " that"
Merge 20:  (l, l)    → "ll"    "will", "all", "well"
```

After 100 merges on WikiText:
- **Vocabulary size**: 165 tokens (65 base chars + 100 learned merges)
- **Encode "hello"**: `[95, 77, 116, 123]` → decode back → `"hello"` ✅

---

## Vocabulary Construction

For character-level tokenization:

```python
chars = sorted(list(set(text)))
```

Each step serves a specific purpose:

| Step | Operation | Why |
|------|-----------|-----|
| `set()` | Remove duplicates | Each character appears once |
| `list()` | Enable indexing | Allows enumeration |
| `sorted()` | Deterministic order | Same vocab every run |

For BPE: the vocabulary is built from base characters **plus** all learned merge tokens.

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

Complete round-trip:

```
Text → encode() → [IDs] → Model → [IDs] → decode() → Text
```

---

## Dataset Creation

The model does not learn from entire texts at once. Every sequence is sliced into input/target pairs:

```python
x = data[i : i + block_size]        # input
y = data[i+1 : i + block_size + 1]  # target (shifted by 1)
```

**Concrete example (block_size=4):**

```
data = [h, e, l, l, o, ' ', w, o, r, l, d]

Slice i=0:
  x = [h, e, l, l]
  y = [e, l, l, o]

  Each position independently teaches:
  h → e
  e → l
  l → l
  l → o
```

From a single text file, this creates **millions** of training examples.

---

## Block Size — The Context Window

```
block_size = 64  (v2)
```

The model can look at the previous 64 tokens when predicting the next one:

```
Tokens: [t₀, t₁, t₂, ..., t₆₃]
                               ↓
                        predict t₆₄
```

| `block_size` | Context | Memory Cost |
|---|---|---|
| `8` | 8 tokens | Minimal |
| `64` | 64 tokens | Low (R1GPT v2) |
| `1024` | 1024 tokens | Medium (GPT-2) |
| `2048` | 2048 tokens | Large (GPT-3) |
| `128K` | 128K tokens | Very large (GPT-4) |

> Context length is one of the primary factors in model quality. Every doubling quadruples the attention memory cost (O(T²)).

---

## Batching

Instead of one sequence per update:

```
1 sequence → 1 gradient update  (slow, high-variance)
```

We process many in parallel:

```
batch_size = 32  (v2)

(B, T) = (32, 64)
↓
32 independent sequences × 64 tokens each
processed simultaneously on GPU/CPU
```

**Benefits:**
- ⚡ Full utilization of hardware parallelism
- 📊 More stable, lower-variance gradient estimates
- 🔄 Faster training per wall-clock second

---

## Embeddings

Raw token IDs carry no semantic meaning:

```
"cat" = 15,  "dog" = 42,  "kitten" = 89
These numbers say nothing about similarity
```

Embeddings fix this with a **learnable lookup table**:

```
Token ID 15
   ↓
Embedding Layer  (learnable, shape: vocab_size × 128)
   ↓
[0.12, -0.44, 1.02, 0.77, ..., -0.31]   ← 128-dim float vector
```

Over training, vectors for similar concepts cluster together in embedding space:

```
cos_similarity("cat", "kitten") → high
cos_similarity("cat", "Paris")  → low
```

R1GPT uses two embedding types:

| Embedding | Table Shape | Encodes |
|---|---|---|
| Token (`tok_emb`) | vocab_size × 128 | What token? |
| Position (`pos_emb`) | block_size × 128 | Where in sequence? |

These are added together: `x = tok_emb + pos_emb`

---

## Why Bigram Models Fail

The first model built was a simple Bigram:

```
Current Token → Predict Next Token  (only 1 token of context)
```

**Problem:** No memory of prior context.

```
"bank" could mean:
  ├─ "river bank"    (geography) — if preceded by "river"
  └─ "bank account"  (finance)   — if preceded by "savings"

Bigram model sees only "bank" → cannot distinguish
```

This limitation directly motivates **Attention** — the mechanism that gives every token access to the full context window.

---

## Attention — Solving the Memory Problem

```
Before Attention:
  Each token processes only itself.
  Representation of "bank" is the same regardless of context.

After Attention:
  "bank" in "river bank" → attends strongly to "river"
  "bank" in "bank account" → attends strongly to "account"
  Representation is dynamically shaped by context.
```

**Flow:**

```
Input Tokens
   ↓
Q, K, V Projections
   ↓
Attention Scores (Q @ Kᵀ / √d_k)
   ↓
Causal Mask (no future tokens)
   ↓
Softmax → Probabilities
   ↓
Weighted Sum of Values
   ↓
Context-Aware Token Representations
```

---

## Causal Masking — No Cheating

GPT is a **decoder-only** model. It must predict each token using only the tokens that came before it:

```
Predicting position 3:
  ✅ Can see: tokens 0, 1, 2
  ❌ Must not see: tokens 4, 5, 6, 7
```

Mask visualization:

```
         attend to:
         pos0  pos1  pos2  pos3
  pos0 [  ✓    -∞    -∞    -∞  ]
  pos1 [  ✓     ✓    -∞    -∞  ]
  pos2 [  ✓     ✓     ✓    -∞  ]
  pos3 [  ✓     ✓     ✓     ✓  ]
```

`-∞` → `softmax(-∞) = 0` → invisible to the model.

This is what makes the training objective **autoregressive** — each position predicts its successor without access to the answer.

---

## Dropout — Regularization in v2

Dropout is a regularization technique that randomly silences neurons during training:

```
Without dropout:
  Network may memorize training data
  Val loss >> Train loss → overfitting

With dropout (p=0.2):
  Each neuron has 20% chance of being zeroed per forward pass
  Network cannot rely on any single pathway
  Forces distributed, robust representations
```

R1GPT v2 adds dropout in two places:
1. After the attention output projection
2. After the feed-forward second linear layer

**Result:** Train and validation loss track closely throughout 5000 steps.

---

## Learning Rate Scheduling — StepLR

Fixed learning rates have a tradeoff:
- High LR early → fast progress
- High LR late → overshooting, instability

StepLR solves this by halving the LR every 1000 steps:

```python
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=1000,
    gamma=0.5
)
```

```
Step     LR
────────────────────────────
0        0.001000
1000     0.000500   (÷2)
2000     0.000250   (÷2)
3000     0.000125   (÷2)
4000     0.000063   (÷2)
5000     0.000031   (÷2)
```

This allows the optimizer to make large steps early when far from the optimum, and fine-tune carefully later.

---

## Validation Loss Monitoring

To detect overfitting and monitor generalization, a 90/10 train/val split is used:

```python
n = int(0.9 * len(data))
train_data = data[:n]
val_data   = data[n:]
```

The `estimate_loss()` function computes average loss over 200 batches without updating gradients:

```python
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(200)
        for k in range(200):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out
```

**R1GPT v2 results (close tracking = healthy training):**

```
Step     Train   Val     Gap
─────────────────────────────────
0        5.067   5.066   0.001 ✅
500      2.036   1.970   0.066 ✅
1000     1.823   1.753   0.070 ✅
2000     1.665   1.599   0.066 ✅
3500     1.585   1.523   0.062 ✅
5000     ~1.56   ~1.50   ~0.06 ✅
```

> Small, consistent gap between train and val loss confirms dropout is working.

---

## Training Loop — Complete Picture

```
┌────────────────────────────────────────────────────┐
│                  Training Loop (×5000)             │
│                                                    │
│  get_batch("train") → x (B,T), y (B,T)            │
│       ↓                                            │
│  model(x, y) → logits (B,T,vocab), loss (scalar)  │
│       ↓                                            │
│  optimizer.zero_grad()   ← clear old gradients     │
│       ↓                                            │
│  loss.backward()         ← compute new gradients   │
│       ↓                                            │
│  optimizer.step()        ← update all weights      │
│       ↓                                            │
│  scheduler.step()        ← decay learning rate     │
│       ↓                                            │
│  if iter % 500 == 0:                               │
│      losses = estimate_loss()                      │
│      print(train loss, val loss, lr)               │
│                                                    │
└────────────────────────────────────────────────────┘
       ↓  after 5000 steps
torch.save(model.state_dict(), "r1gpt_v2.pt")
```

**Optimizer:** `AdamW` — combines Adam's adaptive learning rates with weight decay (L2 regularization on parameters). Standard for modern Transformer training.

---

## Text Generation

Training and generation are **fundamentally different processes**:

```
Training:
  Input + Target → Forward pass → Loss → Backward pass → Update weights
  (supervised, both input and target known)

Generation:
  Input → Forward pass → Sample → Append → Repeat
  (unsupervised, no target, no backward pass)
```

### Top-K Sampling with Temperature

R1GPT v2 uses top-k sampling with temperature for generation:

```python
def generate(self, idx, max_new_tokens):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]
        logits, _ = self(idx_cond)
        logits = logits[:, -1, :]       # last position

        temperature = 0.5               # ← controls randomness
        logits = logits / temperature

        k = 10
        v, ix = torch.topk(logits, k)  # ← keep only top 10
        out = torch.full_like(logits, float('-inf'))
        out.scatter_(1, ix, v)

        probs = torch.softmax(out, dim=-1)
        idx_next = torch.multinomial(probs, 1)
        idx = torch.cat([idx, idx_next], dim=1)
    return idx
```

**Temperature effect:**

```
temperature = 2.0:  flat distribution → very random, creative but incoherent
temperature = 1.0:  raw model distribution → balanced
temperature = 0.5:  sharpened distribution → more focused, repetitive
temperature → 0:    argmax → always pick the most likely token (greedy)
```

**Top-K effect:**

```
k = 1:  always pick the best → greedy, deterministic
k = 10: sample from top 10 → controlled diversity (R1GPT setting)
k = vocab_size: full distribution → maximum randomness
```

---

## The Entire Forward Pass

```
Input text: "The quick"
   ↓  encode()
[42, 18, 91, 7, 63]       ← token IDs (BPE)
   ↓
tok_emb = embedding_table[ids]    (5, 128)
pos_emb = pos_table[0:5]          (5, 128)
x = tok_emb + pos_emb             (5, 128)
   ↓
Block 1: x = x + MHA(LN(x)); x = x + FFWD(LN(x))
Block 2: x = x + MHA(LN(x)); x = x + FFWD(LN(x))
Block 3: x = x + MHA(LN(x)); x = x + FFWD(LN(x))
Block 4: x = x + MHA(LN(x)); x = x + FFWD(LN(x))
   ↓
x = LayerNorm(x)                  (5, 128)
   ↓
logits = lm_head(x)               (5, vocab_size)  ← 283 chars
   ↓
Next token distribution over vocabulary
```

---

## What WikiText Taught the Model

After 5000 training steps on 13.1M characters of Wikipedia-style text:

**Learned patterns:**
```
 = =                           ← WikiText heading style
 The Criship 's lear ...        ← Wikipedia-style prose (partially learned)
 = = = = = = =                 ← Section headings
 When the twing the tale ...    ← Sentence structure
```

**What works:**
- ✅ Sentence-level structure (capital letters, periods, commas)
- ✅ Common English word fragments ("the", "in", "a", "of")
- ✅ WikiText-specific formatting (= headings =)

**What needs improvement:**
- ❌ Proper nouns (invented nonsense words)
- ❌ Long-range coherence
- ❌ Factual accuracy

These limitations stem from the small model size (~873K parameters vs GPT-2's 117M+) and limited training.

---

## Final Outcome — R1GPT v2

| Component | v1 | v2 |
|---|---|---|
| Character Tokenization | ✅ | ✅ |
| BPE Tokenizer | — | ✅ |
| Vocabulary Construction | ✅ | ✅ |
| Dataset Batching | ✅ | ✅ |
| Train/Val Split | — | ✅ |
| Self-Attention Head | ✅ | ✅ |
| Multi-Head Attention | ✅ | ✅ |
| Feed-Forward Network | ✅ | ✅ |
| Dropout Regularization | — | ✅ |
| Residual Connections | ✅ | ✅ |
| Layer Normalization | ✅ | ✅ |
| Transformer Block | ✅ | ✅ (×4) |
| Positional Embeddings | ✅ | ✅ |
| Cross-Entropy Loss | ✅ | ✅ |
| AdamW Optimizer | ✅ | ✅ |
| Learning Rate Scheduler | — | ✅ (StepLR) |
| Validation Loss Monitoring | — | ✅ |
| Model Checkpointing | — | ✅ |
| Temperature + Top-K Sampling | — | ✅ |
| Autoregressive Generation | ✅ | ✅ |

> **Result: A complete, production-style GPT training pipeline — implemented from scratch in PyTorch, trained on 13M characters of real-world text.**

---

## The Most Important Lesson

Building a GPT from scratch revealed that **large language models are not magic**.

They are composed of understandable components:

```
Input Text
   ↓  BPE Tokenization           (text → integer sequences)
   ↓  Token + Positional Embeddings  (integers → vectors + position)
   ↓  ×4 Transformer Blocks
      ├─ Multi-Head Attention     (tokens communicate via Q, K, V)
      ├─ Feed-Forward Network     (each token processes independently)
      ├─ Layer Normalization      (stabilizes activation scale)
      ├─ Residual Connections     (gradient highway through depth)
      └─ Dropout                  (prevents overfitting)
   ↓  LM Head                    (vectors → vocabulary logits)
   ↓  Cross-Entropy Loss          (compare prediction to target)
   ↓  AdamW + StepLR              (update weights, decay LR)
   ↓  Autoregressive Sampling     (temperature + top-k → text)
```

Every component has a specific, understandable purpose. Understanding them individually makes modern AI systems significantly less mysterious — and the path to scaling them up becomes clear.
