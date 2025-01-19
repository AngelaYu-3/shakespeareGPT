# shakespeareGPT

shakespeareGPT is a transformer language model that generates more Shakespearian-like text.

This project is for educational purposes and follows the course [Neural Networks: Zero to Hero by Andrej Karpathy](https://www.youtube.com/watch?v=PaCmpygFfXo&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=2).

***Source code is commented to demonstrate independent understanding of core concepts as well as for ease of future read-throughs. Additional coding files used during independent learning process are also included. Annotated research papers that implemented models are based on are also included.***
___

## Key Papers That Current Implementation Follow

*need to reread*
- [Attention Is All You Need](https://github.com/AngelaYu-3/shakespeareGPT/blob/main/annotated_papers/attentionIsAllYouNeed.pdf)
___

## Resources
Transformers Explained Article Series that helped me better understand the inner-workings of transformers

*need to reread--especially gain familiarity with multi-head attention*
- [Overview of Functionality](https://towardsdatascience.com/transformers-explained-visually-part-1-overview-of-functionality-95a6dd460452)
- [How It Works, Step-By-Step](https://towardsdatascience.com/transformers-explained-visually-part-2-how-it-works-step-by-step-b49fa4a64f34)
- [Multi-Head Attention, Deep Dive](https://towardsdatascience.com/transformers-explained-visually-part-3-multi-head-attention-deep-dive-1c1ff1024853)
- [Not Just How, But Why They Work](https://towardsdatascience.com/transformers-explained-visually-not-just-how-but-why-they-work-so-well-d840bd61a9d3)

___

## Notes
<img src="other/transformerOverview.png" alt="data5" width="500"/>

- query: word for which we are calculating attention
- key/value: word to which we are paying attention
- want attention score to be high between two words that are more relevant to each other
<img src="other/attentionDiagram.png" alt="data1" width="400"/>

- query, key, and value are vectors with an embedding dimension--if two words are more relevant to each other, thsoe vectors are more aligned
<img src="other/encoderDecoderMatrix.png" alt="data2" width="400"/>
<img src="other/encoderDecoderMatrix.png" alt="data3" width="400"/>

- word vectors are generated based on word embedding and weights of linear layers--what is learned by the transformer model
<img src="other/embeddingAndLinear.png" alt="data4" width="400"/>

- self-attention in encoder: source sequence pays attention to itself
- self-attention in decoder: target sequence pays attention to itself
- self-attention in encoder-decoder: target sequence pays attention to source sequences
- positional encoding & word encoding along with ability to process multiple words at once (not sequentially) makes transformers more efficient

___
## Included Dataset

The included example **shakespeareData.txt** dataset has Shakespeare passages [dataset](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt).

___

## License
MIT
