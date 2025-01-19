import torch

# mathematical trick in self-attention (tokens communicating and interacting to predict next token)
B, T, C = 4, 8, 2  # batc, time, channels
x = torch.randn(B, T, C)

# want next token prediction to be based on the weighted average of all the previous tokens
xbow = torch.zeros((B, T, C))
for b in range(8):
    for t in range(T):
        xprev = x[b,:t+1]
        xbow[b,t] = torch.mean(xprev, 0)

# can do the above self-attention token prediction more efficiently using matrix multiplication
a = torch.tril(torch.ones(3, 3))
a = a / torch.sum(a, 1, keepdim=True)
b = torch.randint(0, 10, (3, 2)).float()
c = a@b