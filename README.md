# R1GPT

A GPT-style Transformer built completely from scratch in PyTorch.

## Features

- Character-level tokenization
- Self-Attention
- Multi-Head Attention
- Transformer Blocks
- Autoregressive text generation

## Architecture

```
Input Tokens
↓
Token Embeddings
↓
Positional Embeddings
↓
Transformer Blocks
↓
LayerNorm
↓
LM Head
↓
Next Token Prediction
```

## Learning Goals

This project was built to understand how GPT works internally by implementing every component from first principles.

## Current Status

### R1GPT v1 Specs & Training

R1GPT v1 is a small-scale, character-level generative Transformer model trained to overfit and learn a small 3-line input corpus:
```text
hello world
i love ai
gpt is amazing
```

#### Model Architecture
- **Embedding Dimension (`n_embd`)**: 32
- **Number of Transformer Blocks**: 3
- **Multi-Head Attention**: 4 heads (each head size = 8)
- **Context Size (`block_size`)**: 8 tokens
- **Vocabulary Size**: 17 unique characters
- **Tokenizer**: Character-level

#### Training Configurations
- **Optimizer**: AdamW
- **Learning Rate**: 1e-3
- **Batch Size**: 4
- **Steps**: 1000

#### Loss Log (Sample Run)
- **Step 0**: 3.202
- **Step 100**: 0.865
- **Step 200**: 0.206
- **Step 300**: 0.146
- **Step 400**: 0.095
- **Step 500**: 0.128
- **Step 600**: 0.145
- **Step 700**: 0.109
- **Step 800**: 0.036
- **Step 900**: 0.124
- **Step 1000 (Final Loss)**: ~0.077

#### Generated Sample Output
Starting from a zero-initialized context (`\n`), R1GPT v1 successfully generates:
```text
gpt is amazingpt is amazingpt is amazinghpt is amazingpt is amazingpt is amazingpt is amazingpt is a
```

