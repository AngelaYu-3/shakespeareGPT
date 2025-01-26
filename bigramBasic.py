import torch
import train
import torch.nn as nn
from torch.nn import functional as F

class BigramBasicLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # creating an embedding table to allow for reading of next token based off of given token
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        # predicting what comes next based on a input token
        logits = self.token_embedding_table(idx)  # returns (batch, time, channel) tensor

        # calculating loss
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self(idx)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx


m = BigramBasicLanguageModel(train.vocab_size)  
# PyTorch optimizer--more powerful gradient descent
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)
batch_size = 32
for steps in range(10000):
    xb, yb = train.get_batch('train', batch_size)
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    print('step{}: {}'.format(steps, loss.item()))
# print(logits.shape)
# print(loss)

idx = torch.zeros((1, 1), dtype=torch.long)
print(train.decode(m.generate(idx, max_new_tokens=300)[0].tolist()))

