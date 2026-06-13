import torch

torch.manual_seed(1337)

x = torch.randn(3, 2)

q = x
k = x

wei = q @ k.T

wei = torch.softmax(
    wei,
    dim=-1
)

v = x

out = wei @ v

print(out)

# The key idea: 
# Matrix multiplication allows us to aggregate information 
# from all other positions, weighted by their 
# relevance (softmaxed dot product).