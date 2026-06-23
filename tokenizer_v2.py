import os
import json
import re
from collections import Counter, defaultdict

# ─────────────────────────────────────────
# CONFIG & PATHS
# ─────────────────────────────────────────
dataset_file = "wikitext.txt"
num_merges = 500
vocab_cache_file = "tokenizer_v2_cache.json"

# ─────────────────────────────────────────
# BPE TRAINING / LOADING
# ─────────────────────────────────────────
if os.path.exists(vocab_cache_file):
    print("[Tokenizer] Loading from cache...")
    with open(vocab_cache_file, "r", encoding="utf-8") as f:
        cache = json.load(f)
    # Convert keys back to tuple of strings
    merges = {tuple(k.split(",")): v for k, v in cache["merges"].items()}
    vocab = cache["vocab"]
else:
    print("[Tokenizer] Training BPE on WikiText (500 merges)...")
    if not os.path.exists(dataset_file):
        raise FileNotFoundError(f"Could not find dataset file: {dataset_file}")
        
    with open(dataset_file, "r", encoding="utf-8") as f:
        text = f.read()

    # Split into words/whitespace/punctuation to keep merge logic fast
    word_freqs = Counter(re.findall(r'\w+|\s+|[^\w\s]', text))
    splits = {word: list(word) for word in word_freqs.keys()}

    merges = {}
    for i in range(num_merges):
        pair_freqs = defaultdict(int)
        for word, chars_list in splits.items():
            freq = word_freqs[word]
            for j in range(len(chars_list) - 1):
                pair = (chars_list[j], chars_list[j+1])
                pair_freqs[pair] += freq
        
        if not pair_freqs:
            break
            
        best_pair = max(pair_freqs, key=pair_freqs.get)
        
        new_splits = {}
        for word, chars_list in splits.items():
            new_list = []
            j = 0
            while j < len(chars_list):
                if j < len(chars_list) - 1 and chars_list[j] == best_pair[0] and chars_list[j+1] == best_pair[1]:
                    new_list.append(best_pair[0] + best_pair[1])
                    j += 2
                else:
                    new_list.append(chars_list[j])
                    j += 1
            new_splits[word] = new_list
        splits = new_splits
        merges[best_pair] = best_pair[0] + best_pair[1]

    # Reconstruct vocab
    vocab = set()
    for ch in text:
        vocab.add(ch)
    for token in merges.values():
        vocab.add(token)
    vocab = sorted(list(vocab))

    # Save to cache
    cache = {
        "merges": {f"{k[0]},{k[1]}": v for k, v in merges.items()},
        "vocab": vocab
    }
    with open(vocab_cache_file, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=4)
    print("[Tokenizer] Training complete and cached.")

# Create vocabulary mappings
vocab_size = len(vocab)
stoi = {token: i for i, token in enumerate(vocab)}
itos = {i: token for token, i in stoi.items()}

# ─────────────────────────────────────────
# ENCODE & DECODE
# ─────────────────────────────────────────
word_cache = {}

def encode_word(word, merges_dict):
    if word in word_cache:
        return word_cache[word]
    chars_list = list(word)
    for pair in merges_dict:
        new_list = []
        j = 0
        while j < len(chars_list):
            if j < len(chars_list) - 1 and chars_list[j] == pair[0] and chars_list[j+1] == pair[1]:
                new_list.append(pair[0] + pair[1])
                j += 2
            else:
                new_list.append(chars_list[j])
                j += 1
        chars_list = new_list
    word_cache[word] = chars_list
    return chars_list

def encode(text):
    words = re.findall(r'\w+|\s+|[^\w\s]', text)
    encoded_ids = []
    for w in words:
        tokens_in_word = encode_word(w, merges)
        encoded_ids.extend([stoi[t] for t in tokens_in_word if t in stoi])
    return encoded_ids

def decode(ids):
    tokens = [itos[i] for i in ids if i in itos]
    return "".join(tokens)

# ─────────────────────────────────────────
# SCRIPT VERIFICATION
# ─────────────────────────────────────────
if __name__ == "__main__":
    print("\n--- Tokenizer Check ---")
    print(f"Vocab size (expected 700-1000+): {vocab_size}")
    
    test_str = "machine learning"
    encoded = encode(test_str)
    decoded = decode(encoded)
    
    print(f"\nTest string: '{test_str}'")
    print(f"Encoded IDs: {encoded}")
    print(f"Decoded string: '{decoded}'")
    assert decoded == test_str, "Round-trip encode/decode failed!"
    print("Round-trip validation: PASSED")