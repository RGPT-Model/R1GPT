import time
import re
from collections import Counter, defaultdict

start_time = time.time()

# 1. Load corpus
dataset_file = "wikitext.txt"
with open(dataset_file, "r", encoding="utf-8") as f:
    text = f.read()

# 2. Count word frequencies
word_freqs = Counter(re.findall(r'\w+|\s+|[^\w\s]', text))

# Represent each word as a list of characters
splits = {word: list(word) for word in word_freqs.keys()}

merges = {}
num_merges = 500

for i in range(num_merges):
    # Count pair frequencies
    pair_freqs = defaultdict(int)
    for word, chars_list in splits.items():
        freq = word_freqs[word]
        for j in range(len(chars_list) - 1):
            pair = (chars_list[j], chars_list[j+1])
            pair_freqs[pair] += freq
            
    if not pair_freqs:
        break
        
    best_pair = max(pair_freqs, key=pair_freqs.get)
    
    # Merge the best pair in splits
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

end_time = time.time()
print(f"Training completed in {end_time - start_time:.2f} seconds.")
print(f"Number of merges: {len(merges)}")

# Compute vocabulary
vocab = set()
for ch in text:
    vocab.add(ch)
for token in merges.values():
    vocab.add(token)
vocab = sorted(vocab)
print(f"Vocab size: {len(vocab)}")
