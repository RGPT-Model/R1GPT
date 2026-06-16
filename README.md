# R1GPT

> A GPT-style Transformer language model built completely from scratch in PyTorch — every component implemented by hand, from the BPE tokenizer to autoregressive text generation with temperature-controlled sampling.

---

## What Is This?

R1GPT is a from-scratch implementation of the GPT (Generative Pre-trained Transformer) architecture. The goal was not just to run a model, but to **understand every line of code** — how tokens flow through the network, how attention computes relationships, and how the model learns to generate text.

This is **v2** — trained on WikiText (13M characters) with a full BPE tokenizer pipeline, dropout regularization, learning rate scheduling, and checkpoint saving.

---

## Architecture Overview

```
                         Input Text
                             │
                    ┌────────▼────────┐
                    │  BPE Tokenizer  │   tokenizer_v2.py
                    │  (100 merges)   │
                    └────────┬────────┘
                             │ token IDs
                    ┌────────▼────────┐
                    │ Token Embedding │   (vocab_size × 128)
                    │      +          │
                    │ Pos. Embedding  │   (block_size × 128)
                    └────────┬────────┘
                             │ (B, T, 128)
              ┌──────────────▼──────────────┐
              │      Transformer Block ×4    │
              │                             │
              │  ┌─ LayerNorm               │
              │  │       ↓                  │
              │  │  MultiHeadAttention      │
              │  │  (4 heads × 32 dim)      │
              │  │       ↓                  │
              │  │  Dropout(0.2)            │
              │  └─ + Residual Connection   │
              │                             │
              │  ┌─ LayerNorm               │
              │  │       ↓                  │
              │  │  FeedForward             │
              │  │  128 → 512 → 128         │
              │  │       ↓                  │
              │  │  Dropout(0.2)            │
              │  └─ + Residual Connection   │
              └──────────────┬──────────────┘
                             │
                    ┌────────▼────────┐
                    │  Final LayerNorm │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │     LM Head     │   Linear(128 → vocab_size)
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  Logits         │
                    │  ↓  Training    │   Cross-Entropy Loss
                    │  ↓  Inference   │   Softmax + Top-K + Sample
                    └─────────────────┘
```

---

## Model Specifications

### v2 — WikiText (Current)

| Parameter | Value |
|---|---|
| Architecture | Decoder-only Transformer |
| Tokenizer | BPE (100 merges, `tokenizer_v2.py`) |
| Dataset | WikiText (13.1M characters) |
| Vocabulary Size | 165 BPE tokens |
| Char Vocab (raw) | 283 unique characters |
| Embedding Dimension (`n_embd`) | 128 |
| Context Length (`block_size`) | 64 tokens |
| Transformer Blocks | 4 |
| Attention Heads per Block | 4 |
| Head Dimension (`head_size`) | 32 (`n_embd / n_head`) |
| Feed-Forward Hidden Size | 512 (`4 × n_embd`) |
| Normalization | Pre-LayerNorm |
| Regularization | Dropout `p=0.2` |
| Total Parameters | ~873K |

### v1 — Tiny Corpus (Archived)

| Parameter | Value |
|---|---|
| Tokenizer | Character-level |
| Vocabulary Size | 17 unique characters |
| Embedding Dimension | 32 |
| Context Length | 8 tokens |
| Transformer Blocks | 3 |
| Attention Heads | 4 |

---

## Training Configuration

| Setting | v1 | v2 |
|---|---|---|
| Dataset | 3-line corpus (36 chars) | WikiText (13.1M chars) |
| Optimizer | AdamW | AdamW |
| Learning Rate | `1e-3` (fixed) | `1e-3` → StepLR decay |
| LR Scheduler | None | StepLR (`step=1000`, `γ=0.5`) |
| Batch Size | 4 | 32 |
| Training Steps | 1000 | 5000 |
| Dropout | None | 0.2 |
| Checkpoint | None | `r1gpt_v2.pt` |
| Train/Val Split | None | 90% / 10% |

---

## Training Loss Progression — v2 (WikiText)

```
Step     Loss (Train)   Loss (Val)    LR
────────────────────────────────────────────────
Step    0   5.0675        5.0655       0.001000  ████████████████████  random init
Step  500   2.0355        1.9695       0.001000  ████████
Step 1000   1.8231        1.7528       0.000500  ███████   ← LR halved
Step 1500   1.7144        1.6556       0.000500  ██████▌
Step 2000   1.6650        1.5986       0.000250  ██████    ← LR halved
Step 2500   1.6188        1.5594       0.000250  █████▊
Step 3000   1.5973        1.5379       0.000125  █████▋    ← LR halved
Step 3500   1.5852        1.5232       0.000125  █████▌
Step 4000   1.5754        1.5135       0.000063  █████▍    ← LR halved
Step 4500   1.5614        1.5043       0.000063  █████▎
Final       1.6086        —            —
```

> Initial loss of `~5.07` reflects the larger WikiText vocabulary (283 raw chars). The model converges to `~1.50` validation loss, demonstrating clear and consistent learning across all 5000 steps. Training and validation loss track each other closely, indicating minimal overfitting thanks to dropout.

---

## Generated Output

Starting from a zero context token, the model generates (temperature `0.5`, top-k `10`):

```
 = =


 The Criship 's lear conterning a manages in the season in the singull .

 = = = = = = =



 When the twing the tale asker of the male as a series that it a such concere . As on the al
```

The model has learned:
- **WikiText-style headings** (`= = ... = =`)
- **Sentence structure** (capital letters, periods, commas)
- **Common English word fragments** (`the`, `in`, `a`, `of`)

---

## BPE Tokenizer

The custom Byte-Pair Encoding tokenizer (`tokenizer_v2.py`) learns subword merge rules from the training corpus.

### First 20 Learned Merges (WikiText)

| Merge # | Pair | Result | Why frequent? |
|---|---|---|---|
| 1 | `(e, ' ')` | `e ` | End of words: "the ", "are " |
| 2 | `(t, h)` | `th` | "the", "that", "this" |
| 3 | `(t, ' ')` | `t ` | End of words: "not ", "but " |
| 4 | `(s, ' ')` | `s ` | Plural/verb endings |
| 5 | `(d, ' ')` | `d ` | Past tense endings |
| 6 | `(,, ' ')` | `, ` | Comma + space pattern |
| 7 | `(o, u)` | `ou` | "out", "our", "you" |
| 8 | `(e, r)` | `er` | "other", "over", "after" |
| 9 | `(i, n)` | `in` | "in", "into", "this" |
| 10 | `(y, ' ')` | `y ` | "they ", "by ", "only " |
| 18 | `(' ', th)` | ` th` | " the", " this", " that" |
| 20 | `(l, l)` | `ll` | "will", "all", "well" |

### Encode/Decode Pipeline

```
"hello"
   ↓  encode()
[95, 77, 116, 123]
   ↓  decode()
"hello"
```

---

## Key Concepts Implemented

### Self-Attention Head

Each attention head computes scaled dot-product attention:

```
Attention(Q, K, V) = softmax( QKᵀ / √dₖ ) × V
```

- **Q** — Query: what this token is searching for
- **K** — Key: what this token advertises
- **V** — Value: what this token transmits if selected
- **Causal mask** — lower-triangular, prevents attending to future tokens

### Causal Mask Visualization

```
Position:   0    1    2    3
         ┌────┬────┬────┬────┐
    0    │  ✓ │ -∞ │ -∞ │ -∞ │
    1    │  ✓ │  ✓ │ -∞ │ -∞ │
    2    │  ✓ │  ✓ │  ✓ │ -∞ │
    3    │  ✓ │  ✓ │  ✓ │  ✓ │
         └────┴────┴────┴────┘
  Token i can only attend to tokens ≤ i
```

### Multi-Head Attention

4 attention heads run in parallel, each with `head_size = 32`:

```
Input (B, T, 128)
       │
   ┌───┴───────────────┐
   │  Split across heads│
   └───┬───┬───┬───┬───┘
       H1  H2  H3  H4     each (B, T, 32)
       │   │   │   │
       └───┴───┴───┘
           │  Concat → (B, T, 128)
           │  Linear projection
           ▼
     Output (B, T, 128)
```

### Feed-Forward Network (v2, with Dropout)

```python
self.net = nn.Sequential(
    nn.Linear(128, 512),    # expand 4×
    nn.ReLU(),
    nn.Linear(512, 128),    # compress back
    nn.Dropout(0.2),        # regularize
)
```

### Learning Rate Schedule

```
LR
1e-3 ─────────────┐
                   │ ÷2
5e-4               └─────────┐
                              │ ÷2
2.5e-4                        └─────────┐
...                                      └── ...
     0        1000       2000       3000
                     Steps
```

---

## Project Structure

```
R1GPT/
│
├── head.py              # Full GPT v2: architecture + training loop
│                        # Head, MultiHeadAttention, FeedForward,
│                        # Block, GPTLanguageModel, estimate_loss(),
│                        # StepLR scheduler, checkpoint saving
│
├── tokenizer_v2.py      # Custom BPE tokenizer pipeline
│                        # get_stats(), merge(), encode(), decode()
│                        # Configurable dataset via dataset_file var
│
├── main.py              # Original Bigram baseline model
│
├── scratch.py           # Attention prototyping experiments
│
├── shakespeare.txt      # Shakespeare corpus (1.1M chars)
│
├── wikitext.txt         # WikiText corpus (13.1M chars)
│
├── r1gpt_v2.pt          # Saved model checkpoint
│
├── requirements.txt     # Dependencies (torch)
│
├── .gitignore
│
└── docs/
    ├── attention.md         # Deep-dive: attention math and causal masking
    ├── transformer-block.md # Deep-dive: block architecture and Pre-LN
    └── learning-notes.md    # Complete learning journal (all concepts)
```

---

## Documentation

The `docs/` folder contains detailed technical write-ups:

| File | Contents |
|------|----------|
| [`attention.md`](docs/attention.md) | Attention formula, scaling, causal masking, multi-head design, dropout |
| [`transformer-block.md`](docs/transformer-block.md) | Block structure, Pre-LN vs Post-LN, FFWD network, residual connections |
| [`learning-notes.md`](docs/learning-notes.md) | Full learning journal covering every concept — tokenization to generation |

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

**4. Run BPE tokenizer training**

```bash
python tokenizer_v2.py
```

Edit `dataset_file` at the top of `tokenizer_v2.py` to switch datasets:

```python
dataset_file = "wikitext.txt"    # 13M chars
# dataset_file = "shakespeare.txt"  # 1.1M chars
```

**5. Train the GPT model**

```bash
python head.py
```

Loss is reported every 500 steps. Generated text is printed at the end.

---

## Learning Goals

This project was built to understand how GPT works **from the ground up**, not just to run a pre-trained model:

- ✅ How raw text becomes token IDs (character and BPE tokenization)
- ✅ How BPE merges learn subword structure from a corpus
- ✅ How embeddings give tokens semantic meaning
- ✅ How self-attention enables tokens to communicate
- ✅ How causal masking enforces autoregressive prediction
- ✅ How residual connections stabilize deep networks
- ✅ How LayerNorm normalizes activation scales
- ✅ How dropout prevents overfitting
- ✅ How StepLR scheduling improves convergence
- ✅ How the training loop updates model weights
- ✅ How temperature and top-k control generation diversity

---

## Version History

| Version | Dataset | Tokenizer | Val Loss | Status |
|---------|---------|-----------|----------|--------|
| v1 | 3-line corpus (36 chars) | Character | ~0.077 | ✅ Complete |
| v2 | WikiText (13.1M chars) | BPE (100 merges) | ~1.504 | ✅ Complete |

---

## References

- [Attention Is All You Need — Vaswani et al. (2017)](https://arxiv.org/abs/1706.03762)
- [Language Models are Unsupervised Multitask Learners — Radford et al. (GPT-2)](https://openai.com/research/language-unsupervised)
- [NanoGPT by Andrej Karpathy](https://github.com/karpathy/nanoGPT)
- [Neural Machine Translation of Rare Words with Subword Units — Sennrich et al. (BPE)](https://arxiv.org/abs/1508.07909)
- [The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html)
