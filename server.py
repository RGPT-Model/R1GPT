"""
server.py — R1GPT v1 Backend
Trains the model on startup, then serves inference via /generate.
Run: python server.py
"""

import os
import torch
import torch.nn as nn
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
CORPUS_PATH = os.path.join(os.path.dirname(__file__), "input.txt")
TRAIN_STEPS = 1000
BATCH_SIZE  = 4
BLOCK_SIZE  = 8
N_EMBD      = 32
N_HEAD      = 4
LR          = 1e-3

# ─────────────────────────────────────────
# DATA
# ─────────────────────────────────────────
with open(CORPUS_PATH, "r", encoding="utf-8") as f:
    text = f.read()

chars      = sorted(list(set(text)))
vocab_size = len(chars)
stoi       = {ch: i for i, ch in enumerate(chars)}
itos       = {i: ch for i, ch in enumerate(chars)}

def encode(s):
    return [stoi[c] for c in s if c in stoi]

def decode(ids):
    return "".join(itos[i] for i in ids)

data = torch.tensor(encode(text), dtype=torch.long)

def get_batch():
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    x  = torch.stack([data[i : i + BLOCK_SIZE]     for i in ix])
    y  = torch.stack([data[i + 1 : i + BLOCK_SIZE + 1] for i in ix])
    return x, y

# ─────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key   = nn.Linear(N_EMBD, head_size, bias=False)
        self.query = nn.Linear(N_EMBD, head_size, bias=False)
        self.value = nn.Linear(N_EMBD, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = torch.softmax(wei, dim=-1)
        return wei @ v


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj  = nn.Linear(num_heads * head_size, N_EMBD)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.proj(out)


class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_EMBD, 4 * N_EMBD),
            nn.ReLU(),
            nn.Linear(4 * N_EMBD, N_EMBD),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        head_size = N_EMBD // N_HEAD
        self.sa   = MultiHeadAttention(N_HEAD, head_size)
        self.ffwd = FeedForward()
        self.ln1  = nn.LayerNorm(N_EMBD)
        self.ln2  = nn.LayerNorm(N_EMBD)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table    = nn.Embedding(vocab_size, N_EMBD)
        self.position_embedding_table = nn.Embedding(BLOCK_SIZE, N_EMBD)
        self.blocks = nn.Sequential(Block(), Block(), Block())
        self.ln_f   = nn.LayerNorm(N_EMBD)
        self.lm_head = nn.Linear(N_EMBD, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = nn.functional.cross_entropy(
                logits.view(B * T, C), targets.view(B * T)
            )
        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=1.0):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -BLOCK_SIZE:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            probs  = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
        return idx


# ─────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────
print(f"\n{'='*50}")
print(f"  R1GPT v1 — Training on corpus: '{text.strip()}'")
print(f"  vocab_size={vocab_size}  n_embd={N_EMBD}  n_head={N_HEAD}  block_size={BLOCK_SIZE}")
print(f"  Training for {TRAIN_STEPS} steps...")
print(f"{'='*50}\n")

model     = GPTLanguageModel()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

for step in range(TRAIN_STEPS):
    xb, yb = get_batch()
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    if step % 200 == 0:
        print(f"  step {step:4d} | loss {loss.item():.4f}")

final_loss = loss.item()
print(f"\n  Final loss: {final_loss:.4f}")
print(f"\n{'='*50}")
print(f"  Model ready. Starting server at http://localhost:5000")
print(f"{'='*50}\n")

model.eval()

# ─────────────────────────────────────────
# FLASK APP
# ─────────────────────────────────────────
app = Flask(__name__, static_folder=".", static_url_path="")
CORS(app)


@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/generate", methods=["POST"])
def generate():
    data_req    = request.get_json(force=True)
    seed        = data_req.get("seed", "")
    max_tokens  = int(data_req.get("max_tokens", 100))
    temperature = float(data_req.get("temperature", 1.0))

    # Encode seed — skip unknown chars
    seed_ids = encode(seed)
    if not seed_ids:
        seed_ids = [0]  # fallback to first token

    context = torch.tensor([seed_ids], dtype=torch.long)

    with torch.no_grad():
        output_ids = model.generate(context, max_new_tokens=max_tokens, temperature=temperature)

    generated_text = decode(output_ids[0].tolist())

    return jsonify({
        "output":           generated_text,
        "tokens_generated": max_tokens,
        "final_loss":       round(final_loss, 4),
        "vocab_size":       vocab_size,
        "corpus":           text.strip(),
    })


@app.route("/status")
def status():
    return jsonify({
        "model":       "R1GPT v1",
        "vocab_size":  vocab_size,
        "n_embd":      N_EMBD,
        "n_head":      N_HEAD,
        "block_size":  BLOCK_SIZE,
        "n_blocks":    3,
        "train_steps": TRAIN_STEPS,
        "final_loss":  round(final_loss, 4),
        "corpus":      text.strip(),
        "chars":       chars,
    })


if __name__ == "__main__":
    app.run(debug=False, port=5000)
