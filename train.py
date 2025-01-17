import torch

with open('shakespeareData.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# print("length of dataset in characters: ", len(text))
# print(text[:1000])

chars = sorted(list(set(text)))
vocab_size = len(chars)
# print(''.join(chars))
# print(vocab_size)


"""
create a mapping from characters to integers (tokenizer)
"""
stoi = { ch:i for i, ch in enumerate(chars) }
itos = { i:ch for i, ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]   # encoder: take a string, output a list of integers by encoding each character
decode = lambda l: ''.join([itos[i] for i in l])  # decoder: take a list of integers, output a string by decoding each character
# print(encode("hii there"))
# print(decode((encode("hii there"))))


"""
encode training set, what data will look like to GPT
"""
data = torch.tensor(encode(text), dtype=torch.long)
# print(data.shape, data.dtype)
# print(data[:1000])


"""
split up data into train and validation sets
"""
n = int(0.9*len(data))  # first 90% is train, 10% is val
train_data = data[:n]
val_data = data[n:]


"""
create training mini-batches
"""
#batch_size = 4   # how many independent sequences will be processed in parallel
block_size = 8   # the maximum context length for predictions

# generate a small batch of data of inputs x and targets y
def get_batch(split, batch_size):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:block_size+i+1] for i in ix])
    return x, y




