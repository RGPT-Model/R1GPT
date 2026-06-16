import torch
import torch.nn as nn

with open("wikitext.txt", "r", encoding="utf-8") as f:
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
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

batch_size = 32
block_size = 64
max_iters = 5000

def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

n_embd = 128
n_head = 4
n_layer = 4
dropout = 0.2
head_size = n_embd // n_head

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
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat(
            [h(x) for h in self.heads],
            dim=-1
        )

        out = self.proj(out)
        out = self.dropout(out)
        return out

mha = MultiHeadAttention(
    num_heads=n_head,
    head_size=head_size
)

x = torch.randn(4, 8, n_embd)

out = mha(x)

print(out.shape)

class FeedForward(nn.Module):

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

# Test
ffwd = FeedForward()

x = torch.randn(4, 8, n_embd)

out = ffwd(x)

print(out.shape)

class Block(nn.Module):

    def __init__(self, n_embd=n_embd, n_head=n_head):
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

x = torch.randn(4, 8, n_embd)

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
            *[Block() for _ in range(n_layer)]
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
            temperature = 0.5
            logits = logits / temperature
            k = 10
            v, ix = torch.topk(logits, k)
            out = torch.full_like(logits, float('-inf'))
            out.scatter_(1, ix, v)
            probs = torch.softmax(out, dim=-1)
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

print(get_batch("train")[0].shape)
print(get_batch("val")[0].shape)

model = GPTLanguageModel(vocab_size=vocab_size)
# model.load_state_dict(
#     torch.load("r1gpt_v2.pt")
# )
print(model)
print(
    sum(
        p.numel()
        for p in model.parameters()
    )
)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(200)
        for k in range(200):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3
)
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=1000,
    gamma=0.5
)

for iter in range(max_iters):

    xb, yb = get_batch("train")

    logits, loss = model(
        xb,
        yb
    )

    optimizer.zero_grad(set_to_none=True)

    loss.backward()

    optimizer.step()
    scheduler.step()

    if iter % 500 == 0:
        losses = estimate_loss()
        print(
            f"step {iter}: "
            f"train loss {losses['train']:.4f}, "
            f"val loss {losses['val']:.4f}, "
            f"lr {scheduler.get_last_lr()[0]:.6f}"
        )

print(loss.item())

torch.save(
    model.state_dict(),
    "r1gpt_v2.pt"
)

context = torch.zeros(
    (1,1),
    dtype=torch.long
)

generated = model.generate(
    context,
    max_new_tokens=200
)

print(
    decode(
        generated[0].tolist()
    )
)
