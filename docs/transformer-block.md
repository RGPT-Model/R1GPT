# Transformer Block in R1GPT

The `Block` is the fundamental modular unit of the Transformer architecture. It stacks multi-head self-attention and position-wise feed-forward networks, connected with residual connections and layer normalization.

## 1. Structure of a Block

Each Transformer block in R1GPT is comprised of:
1. **Multi-Head Attention (MHA)**: Allows token representation aggregation across the sequence.
2. **Feed-Forward Network (FFWD)**: Acts as a position-wise MLP that processes token representation independently.
3. **Layer Normalization (LayerNorm)**: Stabilizes training dynamics by normalizing features.
4. **Residual Connections**: Crucial path for uninhibited gradient flow during backpropagation.

## 2. Pre-LN vs. Post-LN

Original Transformer models (Attention Is All You Need) used **Post-LayerNorm**:
$$ x_{next} = \text{LayerNorm}(x + \text{SubLayer}(x)) $$

Modern architectures (including GPT-2, GPT-3, and R1GPT) adopt **Pre-LayerNorm**:
$$ x_{next} = x + \text{SubLayer}(\text{LayerNorm}(x)) $$

### Why Pre-LN?
By applying LayerNorm *before* passing the representation to the self-attention or feed-forward layer, the input to each layer is properly scaled. Crucially, the residual path is kept clear, allowing gradients to propagate directly from the output layer to the input embeddings without going through any LayerNorm operations. This enables training deep architectures without vanishing/exploding gradients.

## 3. Feed-Forward Network (FFWD)

Once attention has aggregated information from other tokens, the Feed-Forward network processes each token position independently. 

It consists of:
- A linear layer projecting `n_embd` to `4 * n_embd`.
- A non-linear activation function (ReLU in our current implementation, though models like GPT-2 use GELU).
- A linear layer projecting `4 * n_embd` back to `n_embd`.

```python
self.net = nn.Sequential(
    nn.Linear(n_embd, 4 * n_embd),
    nn.ReLU(),
    nn.Linear(4 * n_embd, n_embd)
)
```
This projection allows the model to learn complex non-linear combinations of the attended features.
