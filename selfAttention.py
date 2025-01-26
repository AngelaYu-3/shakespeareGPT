import torch
import torch.nn as nn
from torch.nn import functional as F

"""
attention is a communication mechanism--can be visualized as nodes (where each node has a vector of information) in a directed graph 
looking at each other and aggregating with a weighted sum from all nodes that point to them, with data-dependent weights

in principle, attention can be used in any directed graph, just so happens to be a very useful way of finding relationships for autoregressive language models

there is no notion of space--attention simply acts over a set of vectors (vectors don't know where they are in space), 
which is why we need to positionally encode tokens so vectors know where they are

each example across a batch dimension is processed completely independently and never "talk" to each other (different batches or "datasets")

in an "encoder" attention block, just delete the single line that does masking, allowing all tokens to communicate (varies case by case)

in a "decoder" attention block, there is triangular masking and is usually used in autoregressive settings so previous nodes don't talk to future nodes (varies case by case)

self-attention (nodes just like to look and talk to each other) means that the keys and values are produced from the same source as queries (same input goes through linear layers), in cross-attention the
key and values are produced from different source as queries (used when there is a separate source of nodes we'd like to pull information from into our noes)

scaled attention divides wei by 1/sqrt(head_size), making it so when input query and keys are unit varience, wei will be unit variance too
and softmax will stay diffused and not saturate too much (as seen in attention equation in "All You Need Is Attention" paper)
"""

# mathematical trick in self-attention (tokens communicating and interacting to predict next token)
B, T, C = 4, 8, 2  # batc, time, channels
x = torch.randn(B, T, C)

# want next token prediction to be based on the weighted average of all the previous tokens
xbow = torch.zeros((B, T, C))
for b in range(B):
    for t in range(T):
        xprev = x[b,:t+1]
        xbow[b,t] = torch.mean(xprev, 0)

# can do the above self-attention token prediction more efficiently using matrix multiplication
# implementing attention equation from "All You Need Is Attention" paper for a single Head performing self-attention
B, T, C = 4, 8, 32  # batch, time, channels
x = torch.randn(B, T, C)

head_size = 16
key = nn.Linear(C, head_size, bias=False)
query = nn.Linear(C, head_size)
value = nn.Linear(C, head_size, bias=False)
k = key(x)
q = query(x)
wei = q @ k.transpose(-2, -1)

tril = torch.tril(torch.ones(T, T))
wei = wei.masked_fill(tril == 0, float('-inf'))
wei = F.softmax(wei, dim=-1)

v = value(x)
out = wei @ v
