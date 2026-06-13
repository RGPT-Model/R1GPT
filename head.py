import torch
import torch.nn as nn

with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

print(len(text))

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }

def encode(s):
    return [stoi[c] for c in s]

def decode(s):
    return "".join([itos[i] for i in s])

data = torch.tensor(encode(text))

batch_size = 4
block_size = 8

def get_batch():
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

n_embd = 32
head_size = 16

class Head(nn.Module):

    def __init__(self, head_size):
        super().__init__()

        self.key = nn.Linear(
            n_embd,
            head_size,
            bias=False
        )

        self.query = nn.Linear(
            n_embd,
            head_size,
            bias=False
        )

        self.value = nn.Linear(
            n_embd,
            head_size,
            bias=False
        )

        self.register_buffer(
            'tril',
            torch.tril(
                torch.ones(block_size, block_size)
            )
        )

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)

        wei = wei.masked_fill(
            self.tril[:T, :T] == 0,
            float('-inf')
        )

        wei = torch.softmax(wei, dim=-1)

        out = wei @ v

        return out

class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, head_size):
        super().__init__()

        self.heads = nn.ModuleList(
            [Head(head_size) for _ in range(num_heads)]
        )
        self.proj = nn.Linear(num_heads * head_size, n_embd)

    def forward(self, x):
        out = torch.cat(
            [h(x) for h in self.heads],
            dim=-1
        )

        out = self.proj(out)
        return out

mha = MultiHeadAttention(
    num_heads=4,
    head_size=16
)

x = torch.randn(4, 8, 32)

out = mha(x)

print(out.shape)

class FeedForward(nn.Module):

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd)
        )

    def forward(self, x):
        return self.net(x)

# Test
ffwd = FeedForward()

x = torch.randn(4,8,32)

out = ffwd(x)

print(out.shape)

class Block(nn.Module):

    def __init__(self, n_embd=n_embd, n_head=4):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

# Test Block
block = Block()

x = torch.randn(4,8,32)

out = block(x)

print(out.shape)

class GPTLanguageModel(nn.Module):

    def __init__(self, vocab_size=65):
        super().__init__()

        self.token_embedding_table = nn.Embedding(
            vocab_size,
            n_embd
        )

        self.position_embedding_table = nn.Embedding(
            block_size,
            n_embd
        )

        self.blocks = nn.Sequential(
            Block(),
            Block(),
            Block()
        )

        self.ln_f = nn.LayerNorm(n_embd)

        self.lm_head = nn.Linear(
            n_embd,
            vocab_size
        )
    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(
            torch.arange(T)
        )

        x = tok_emb + pos_emb

        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = nn.functional.cross_entropy(
                logits,
                targets
            )

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = torch.softmax(
                logits,
                dim=-1
            )
            idx_next = torch.multinomial(
                probs,
                num_samples=1
            )
            idx = torch.cat(
                [
                    idx,
                    idx_next
                ], 
                dim=1
            )
        return idx

model = GPTLanguageModel(vocab_size=vocab_size)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3
)

for step in range(1000):

    xb, yb = get_batch()

    logits, loss = model(
        xb,
        yb
    )

    optimizer.zero_grad(set_to_none=True)

    loss.backward()

    optimizer.step()

    if step % 100 == 0:
        print(step, loss.item())

print(loss.item())

context = torch.zeros(
    (1,1),
    dtype=torch.long
)

generated = model.generate(
    context,
    max_new_tokens=100
)

print(
    decode(
        generated[0].tolist()
    )
)