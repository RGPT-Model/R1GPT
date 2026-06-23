import time
import re
import os
import json
from collections import Counter

start_time = time.time()

# Load cache
vocab_cache_file = "tokenizer_v2_cache.json"
if not os.path.exists(vocab_cache_file):
    print("Run test_bpe.py first to generate cache")
    exit(1)

with open(vocab_cache_file, "r", encoding="utf-8") as f:
    cache = json.load(f)
merges = {tuple(k.split(",")): v for k, v in cache["merges"].items()}
vocab = cache["vocab"]

stoi = {t: i for i, t in enumerate(vocab)}
itos = {i: t for i, t in enumerate(vocab)}

# Create word cache
word_cache = {}
def encode_word(word, merges_list):
    if word in word_cache:
        return word_cache[word]
    chars_list = list(word)
    for pair in merges_list:
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
    encoded_tokens = []
    for w in words:
        encoded_tokens.extend([stoi[t] for t in encode_word(w, merges) if t in stoi])
    return encoded_tokens

def decode(ids):
    tokens = [itos[i] for i in ids]
    return "".join(tokens)

# Load corpus
with open("wikitext.txt", "r", encoding="utf-8") as f:
    text = f.read()

print("Encoding full text...")
ids = encode(text)
print(f"Encoded to {len(ids)} tokens in {time.time() - start_time:.2f} seconds.")

print("Testing round-trip decoding...")
decoded_text = decode(ids)
print("Is decode(encode(text)) == text?", decoded_text == text)

print("Test prompt:")
prompt = "machine learning"
p_ids = encode(prompt)
print(f"encode('{prompt}') = {p_ids}")
print(f"decode({p_ids}) = '{decode(p_ids)}'")
