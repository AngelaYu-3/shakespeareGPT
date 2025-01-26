import torch
import torch.nn as nn
from torch.nn import functional as F

"""
hyperparameters
"""
batch_size = 16  # how many independent sequences are processed in parallel
block_size = 32  # the maximum context length for predictions
max_iters = 10000
eval_interval = 100
learning_rate = 1e-3
eval_iters = 200
n_embd = 64
n_head = 4
n_layer = 4
dropout = 0.0

"""
get data set and prep with character to integer mapping (encoding and decoding)
and create training and validation data
"""
with open('shakespeareData.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i, ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))  # first 90% of data will be for training, the rest if for validation
train_data = data[:n]
val_data = data[n:]


"""
data loading
"""
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y
    

"""
estimate loss
"""
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


"""
one head of self-attention 
"""
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))   # lower triangular matrix for masking

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)

        # compute attention scores using attention equation in "All You Need Is Attention" paper
        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        # perform weighted aggregation of the values
        v = self.value(x)
        out = wei @ v
        return out


"""
multi-head attention
just multiple heads of attention in parallel that are then aggregated
"""
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out
    

"""
simple linear layer followed by a non-linearity
"""
class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )


    def forward(self, x):
        return self.net(x)
    

"""
intersperses communication or attention followed by computation or linear feed forward
"""
class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
    

"""
simple bigram transformer language model
___________________________________________________________________________________________________________________________________________________________________


the bigram transformer model still allows all tokens to attend to each other, but just has more of a focus on predicting adjacent pairs
the model might learn to emphasize the most immediate or local dependencies more heavily in its attention scores, even though it still has access
to the entire previous sequence--simply trained to emphasize adjacent tokens more


bigram transformer pros:
- faster training (primarily focuses on local and adjacent tokens)
- works well for local dependnecies (ie: autocompletion or rudimentary language modeling)
- lower memory usage for systems with limited hardware resources

bigram traansformer cons:
- limited capacity for long-range dependncies because of its primary focus on adjacent tokens (ie: understanding relationships like subject-verb or coreference
where "he" refers to a previously mentioned subject)
- potential for overfitting to short-range patterns due to its "near-sightedness" (ie: missing out on broader, more complex language structures)
- less generalizable--once again due to "near-sightedness"
- difficult to handle ambiguity (ie: understanding different meanings of the same word based on context)
"""



"""
batch--number of samples
time--number of time steps or sequence length (ie: tokens in a sentence)
channel--number of features at each time step (ie: embedding size for each word, number of channels in an image)
"""
class BigramTransformerLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # note that there is a position embedding table for a transformer model
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        """
        self.sa_heads = MultiHeadAttention(4, n_embd//4)  # ie: 4 heads of 8-dimensional self-attention
        self.sa_head = Head(n_embd)  # self attention mechanism, processes each token in the sequence to understand its attention with other tokens
        self.ffwd = FeedForward(n_embd)

        self.blocks = nn.Sequential(
            Block(n_embd, n_head=4),
            Block(n_embd, n_head=4),
            Block(n_embd, n_head=4),
            nn.LayerNorm(n_embd),
        )
        """
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)  # linear layer, maps output of transformer to vocab space

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_embd = self.token_embedding_table(idx)
        pos_embd = self.position_embedding_table(torch.arange(T))
        x = tok_embd + pos_embd
        # x = self.sa_heads(x)  # applying one head of self-attention
        # x = self.ffwd(x)
        x = self.blocks(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens, because we give it a constraint of maximum length for predictions (how much of prior tokens it can use to predict next token)
            idx_cond = idx[:, -block_size:]
            # get the predictions by calling forward function
            logits, loss = self(idx_cond)
            # focus only on the last time step (bigram)
            logits = logits[:, -1, :]
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
    
    
model = BigramTransformerLanguageModel()
# create a pyTorch optimizer--more powerful gradient descent
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
print('starting!')

for iter in range(max_iters):
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        # sample a batch of data
        xb, yb = get_batch('train')

        # evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

# generate from trained model
context = torch.zeros((1, 1), dtype=torch.long)
print(decode(model.generate(context, max_new_tokens=2000)[0].tolist()))








