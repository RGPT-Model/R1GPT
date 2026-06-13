# R1GPT

> A GPT-style Transformer language model built completely from scratch in PyTorch — every component implemented by hand, from the attention mechanism to autoregressive text generation.

---

## What Is This?

R1GPT is a from-scratch implementation of the GPT (Generative Pre-trained Transformer) architecture. The goal was not just to run a model, but to **understand every line of code** — how tokens flow through the network, how attention computes relationships, and how the model learns to generate text.

This is v1 — trained on a small 3-line corpus as a proof-of-concept:

```
hello world
i love ai
gpt is amazing
```

After 1000 training steps, the model learned to generate text that closely follows these patterns.

---

## Architecture Overview

```
Input Characters
       ↓
  Tokenizer (char → int)
       ↓
  Token Embedding Table       (vocab_size × n_embd)
       +
  Positional Embedding Table  (block_size × n_embd)
       ↓
┌──────────────────────────────┐
│       Transformer Block ×3   │
│                              │
│  LayerNorm                   │
│       ↓                      │
│  Multi-Head Attention (4h)   │
│       ↓                      │
│  + Residual Connection       │
│                              │
│  LayerNorm                   │
│       ↓                      │
│  Feed-Forward Network        │
│       ↓                      │
│  + Residual Connection       │
└──────────────────────────────┘
       ↓
  Final LayerNorm
       ↓
  LM Head (Linear: n_embd → vocab_size)
       ↓
  Logits → Cross-Entropy Loss / Softmax + Sample
```

---

## Model Specifications (v1)

| Parameter | Value |
|---|---|
| Architecture | Decoder-only Transformer |
| Tokenizer | Character-level |
| Vocabulary Size | 17 unique characters |
| Embedding Dimension (`n_embd`) | 32 |
| Context Length (`block_size`) | 8 tokens |
| Transformer Blocks | 3 |
| Attention Heads per Block | 4 |
| Head Dimension (`head_size`) | 8 (`n_embd / n_head`) |
| Feed-Forward Expansion | 4× (`128` hidden units) |
| Normalization | Pre-LayerNorm |

---

## Training Configuration

| Setting | Value |
|---|---|
| Training Dataset | 3-line custom corpus (36 chars) |
| Optimizer | AdamW |
| Learning Rate | `1e-3` |
| Batch Size | 4 |
| Training Steps | 1000 |
| Loss Function | Cross-Entropy |

---

## Training Loss Progression

The model started near the theoretical maximum loss for random initialization and steadily converged:

```
Step    0  →  3.202   ████████████████████  (random)
Step  100  →  0.865   ████████
Step  200  →  0.206   ██
Step  300  →  0.146   █▌
Step  400  →  0.095   █
Step  500  →  0.128   █▎
Step  600  →  0.145   █▌
Step  700  →  0.109   █
Step  800  →  0.036   ▌
Step  900  →  0.124   █▎
Step 1000  →  0.077   ▊   (final)
```

> Initial loss of `~3.2` is expected: for a 17-token vocabulary, random prediction gives loss ≈ `ln(17) ≈ 2.83`. The model reaches a final loss of `~0.077`, demonstrating clear and substantial learning.

---

## Generated Output

Starting from a zero-initialized context token, the trained model generates:

```
gpt is amazingpt is amazingpt is amazinghpt is amazingpt is amazingpt is a
```

The model successfully learned the structure and character patterns from the training corpus, demonstrating functional autoregressive generation.

---

## Key Concepts Implemented

### Self-Attention Head

Each attention head computes scaled dot-product attention:

```
Attention(Q, K, V) = softmax( QKᵀ / √dₖ ) × V
```

- **Q** — Query: what each token is looking for
- **K** — Key: what each token contains
- **V** — Value: the information to pass forward
- **Causal mask** prevents attention to future tokens (`-inf` fill → `0` after softmax)

### Multi-Head Attention

4 attention heads run in parallel, each capturing different token relationships:

```python
out = torch.cat([h(x) for h in self.heads], dim=-1)
out = self.proj(out)
```

### Feed-Forward Network

Position-wise MLP applied after each attention block:

```
Linear(32 → 128) → ReLU → Linear(128 → 32)
```

### Pre-LayerNorm Residual Blocks

Modern GPT-style normalization applied *before* each sublayer:

```python
x = x + self.sa(self.ln1(x))    # attention path
x = x + self.ffwd(self.ln2(x))  # feed-forward path
```

### Positional Embeddings

Since Transformers are order-agnostic, position is encoded explicitly:

```python
x = tok_emb + pos_emb   # (B, T, C)
```

---

## Project Structure

```
R1GPT/
│
├── head.py              # Full GPT model: Head, MultiHeadAttention,
│                        # FeedForward, Block, GPTLanguageModel
│
├── main.py              # Original Bigram baseline model
│
├── scratch.py           # Attention prototyping experiments
│
├── input.txt            # Training corpus
│
├── requirements.txt     # Dependencies
│
├── .gitignore
│
└── docs/
    ├── attention.md         # Deep-dive: attention math and causal masking
    ├── transformer-block.md # Deep-dive: block architecture and Pre-LN
    └── learning-notes.md    # Complete learning journal (23 topics)
```

---

## Documentation

The `docs/` folder contains detailed technical writeups:

| File | Contents |
|------|----------|
| [`attention.md`](docs/attention.md) | Attention formula, scaling, causal masking, multi-head design |
| [`transformer-block.md`](docs/transformer-block.md) | Block structure, Pre-LN vs Post-LN, FFWD network |
| [`learning-notes.md`](docs/learning-notes.md) | 23-topic learning journal covering every concept built |

---

## Quickstart

**1. Clone the repository**

```bash
git clone https://github.com/RGPT-Model/R1GPT.git
cd R1GPT
```

**2. Create a virtual environment**

```bash
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # macOS/Linux
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Run training**

```bash
python head.py
```

You will see loss printed every 100 steps, followed by a sample of generated text.

---

## Learning Goals

This project was built to understand how GPT works **from the ground up**, not just to run a pre-trained model:

- ✅ How raw text becomes token IDs
- ✅ How embeddings give tokens semantic meaning
- ✅ How self-attention enables tokens to communicate
- ✅ How causal masking enables autoregressive prediction
- ✅ How residual connections stabilize deep networks
- ✅ How LayerNorm stabilizes activation scales
- ✅ How the training loop updates model weights
- ✅ How text generation emerges from next-token sampling

---

## Why Build From Scratch?

Using a library like HuggingFace `transformers` means you can *use* a GPT model. Building one from scratch means you can *understand* it.

> Every matrix multiplication, every softmax, every residual addition has a reason. R1GPT was built to find those reasons.

---

## Status

| Version | Dataset | Status |
|---------|---------|--------|
| v1 | 3-line corpus (36 chars) | ✅ Complete |
| v2 | Tiny Shakespeare (1.1M chars) | 🔜 Planned |

---

## References

- [Attention Is All You Need — Vaswani et al. (2017)](https://arxiv.org/abs/1706.03762)
- [Language Models are Unsupervised Multitask Learners — Radford et al. (GPT-2)](https://openai.com/research/language-unsupervised)
- [NanoGPT by Andrej Karpathy](https://github.com/karpathy/nanoGPT)
- [The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html)
