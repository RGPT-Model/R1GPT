import sys
import torch
import torch.nn as nn

# ─────────────────────────────────────────
# CONFIG & HYPERPARAMETERS
# ─────────────────────────────────────────
N_EMBD = 128
N_HEAD = 4
N_LAYER = 4
DROPOUT = 0.2
BLOCK_SIZE = 64

# ANSI Colors for premium CLI experience
BLUE = "\033[94m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
MAGENTA = "\033[95m"
WHITE = "\033[97m"
BOLD = "\033[1m"
RESET = "\033[0m"

# ─────────────────────────────────────────
# MODEL DEFINITION
# ─────────────────────────────────────────
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(N_EMBD, head_size, bias=False)
        self.query = nn.Linear(N_EMBD, head_size, bias=False)
        self.value = nn.Linear(N_EMBD, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        
        wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = torch.softmax(wei, dim=-1)
        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads * head_size, N_EMBD)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out


class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_EMBD, 4 * N_EMBD),
            nn.ReLU(),
            nn.Linear(4 * N_EMBD, N_EMBD),
            nn.Dropout(DROPOUT)
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embd=N_EMBD, n_head=N_HEAD):
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


class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, N_EMBD)
        self.position_embedding_table = nn.Embedding(BLOCK_SIZE, N_EMBD)
        self.blocks = nn.Sequential(*[Block() for _ in range(N_LAYER)])
        self.ln_f = nn.LayerNorm(N_EMBD)
        self.lm_head = nn.Linear(N_EMBD, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits, None


@torch.no_grad()
def generate_stream(model, idx, max_new_tokens, temperature=0.6, top_k=10):
    """
    Generator that yields character tokens one by one for streaming print.
    """
    for _ in range(max_new_tokens):
        # crop idx to the last block_size tokens
        idx_cond = idx[:, -BLOCK_SIZE:]
        logits, _ = model(idx_cond)
        # focus only on the last time step
        logits = logits[:, -1, :]
        # apply temperature scaling
        logits = logits / max(temperature, 1e-6)
        
        # optionally crop logits to only top k options
        if top_k is not None:
            v, ix = torch.topk(logits, min(top_k, logits.shape[-1]))
            out = torch.full_like(logits, float('-inf'))
            out.scatter_(1, ix, v)
            probs = torch.softmax(out, dim=-1)
        else:
            probs = torch.softmax(logits, dim=-1)
            
        # sample from the distribution
        idx_next = torch.multinomial(probs, num_samples=1)
        # append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)
        yield idx_next.item()


def main():
    print(f"{CYAN}{BOLD}==================================================")
    print("        R1GPT v2 - Interactive CLI Chat           ")
    print(f"=================================================={RESET}")

    try:
        print(f"{YELLOW}Loading BPE tokenizer...{RESET}", end="", flush=True)
        import tokenizer_v2
        vocab_size = tokenizer_v2.vocab_size
        stoi = tokenizer_v2.stoi
        itos = tokenizer_v2.itos
        encode = tokenizer_v2.encode
        decode = tokenizer_v2.decode
        chars = tokenizer_v2.vocab
        print(f" {GREEN}Done. (Vocab size: {vocab_size}){RESET}")
    except Exception as e:
        print(f"\n{RED}Error loading tokenizer: {e}{RESET}")
        sys.exit(1)

    # Detect device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"{YELLOW}Device selected:{RESET} {BOLD}{device.upper()}{RESET}")

    # Load model
    try:
        print(f"{YELLOW}Loading model architecture...{RESET}", end="", flush=True)
        model = GPTLanguageModel(vocab_size)
        print(f" {GREEN}Done.{RESET}")
        
        print(f"{YELLOW}Loading checkpoint 'r1gpt_v2.pt'...{RESET}", end="", flush=True)
        checkpoint_path = "r1gpt_v2.pt"
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.to(device)
        model.eval()
        print(f" {GREEN}Done. Model ready!{RESET}")
    except Exception as e:
        print(f"\n{RED}Error loading model: {e}{RESET}")
        sys.exit(1)

    print(f"\n{CYAN}{BOLD}--------------------------------------------------")
    print("Instructions:")
    print(" - Type your prompt and press Enter.")
    print(" - The model will autoregressively complete/reply to it.")
    print(" - Type 'exit', 'quit', or '/exit' to quit.")
    print(f"--------------------------------------------------{RESET}\n")

    while True:
        try:
            # Get user prompt
            prompt = input(f"{BOLD}{BLUE}You:{RESET} ")
            
            # Clean and validate input
            if not prompt.strip():
                continue
            
            if prompt.strip().lower() in ["exit", "quit", "/exit"]:
                print(f"\n{MAGENTA}Goodbye!{RESET}")
                break

            # Encode prompt. If empty because of unseen chars, fallback to space
            encoded_prompt = encode(prompt)
            if not encoded_prompt:
                encoded_prompt = encode(" ")

            context = torch.tensor([encoded_prompt], dtype=torch.long, device=device)
            
            print(f"{BOLD}{GREEN}R1GPT:{RESET} ", end="", flush=True)
            
            # Run streaming generation
            char_count = 0
            # Generate up to 250 new tokens
            for token_id in generate_stream(model, context, max_new_tokens=250, temperature=0.6, top_k=10):
                char = itos[token_id]
                print(char, end="", flush=True)
                
                # Append to context so we keep generating
                token_tensor = torch.tensor([[token_id]], dtype=torch.long, device=device)
                context = torch.cat((context, token_tensor), dim=1)
                char_count += 1
                
            print("\n")  # Newline after done
            
        except KeyboardInterrupt:
            print(f"\n\n{MAGENTA}Goodbye! (Session interrupted){RESET}")
            break
        except Exception as e:
            print(f"\n{RED}An error occurred: {e}{RESET}\n")


if __name__ == "__main__":
    main()
