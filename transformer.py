import torch

# mathematical trick in self-attention (tokens communicating and interacting to predict next token)
B, T, C = 4, 8, 2  # batc, time, channels
x = torch.randn(B, T, C)

# want next token prediction to be based on the average of all the previous tokens
xbow = torch.zeros((B, T, C))
for b in range(8):
    for t in range(T):
        xprev = x[b,:t+1]
        xbow[b,t] = torch.mean(xprev, 0)


