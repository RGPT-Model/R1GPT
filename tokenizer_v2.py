from collections import Counter

def get_stats(tokens):

    pairs = Counter()

    for i in range(len(tokens)-1):
        pairs[
            (
                tokens[i],
                tokens[i+1]
            )
        ] += 1

    return pairs

def merge(tokens, pair):

    new_tokens = []

    i = 0

    while i < len(tokens):

        if (
            i < len(tokens) - 1
            and
            tokens[i] == pair[0]
            and
            tokens[i + 1] == pair[1]
        ):

            new_tokens.append(
                pair[0] + pair[1]
            )

            i += 2

        else:

            new_tokens.append(
                tokens[i]
            )

            i += 1

    return new_tokens

dataset_file = "wikitext.txt"

with open(
    dataset_file,
    "r",
    encoding="utf-8"
) as f:

    text = f.read()

tokens = list(text)

merges = {}

# Try 100 merges
for i in range(100):
    pairs = get_stats(tokens)

    best_pair = max(
        pairs,
        key=pairs.get
    )

    print(f"Merge {i+1}: {best_pair}")

    tokens = merge(
        tokens,
        best_pair
    )

    merges[best_pair] = (
        best_pair[0] + best_pair[1]
    )

print("Final merges (first 20):")
print(
    list(merges.items())[:20]
)

vocab = set()

for ch in text:
    vocab.add(ch)

for token in merges.values():
    vocab.add(token)

vocab = sorted(vocab)

print("Vocabulary size:")
print(
    len(vocab)
)

stoi = {
    token: i
    for i, token in enumerate(vocab)
}

itos = {
    i: token
    for token, i in stoi.items()
}

def encode(text):

    tokens = list(text)

    for pair in merges:

        tokens = merge(
            tokens,
            pair
        )

    return [
        stoi[token]
        for token in tokens
    ]

def decode(ids):

    tokens = [
        itos[i]
        for i in ids
    ]

    return "".join(tokens)

ids = encode(
    "hello"
)

print(ids)

print(
    decode(ids)
)