# shakespeareGPT

shakespeareGPT is a transformer language model that generates more Shakespearian-like text.

This project is for educational purposes and follows the course [Neural Networks: Zero to Hero by Andrej Karpathy](https://www.youtube.com/watch?v=PaCmpygFfXo&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=2).

***Source code is commented to demonstrate independent understanding of core concepts as well as for ease of future read-throughs. Additional coding files used during independent learning process are also included. Annotated research papers that implemented models are based on are also included.***
___

## Brief Model Descriptions
- **bigram:** one character predicts the next character with a lookup table of counts from name.txt data
- **bigram_nn:** one character predicts the next character with neural network trained on name.txt data
- **mlp:** a set length of characters predict the next character with a multi-layer perceptron model trained on name.txt data

## Key Papers That Current Implementations Follow
- [A Neural Probablistic Language Model](https://github.com/AngelaYu-3/makemore/blob/main/annotated_papers/MLP_paper.pdf)
- [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://github.com/AngelaYu-3/makemore/blob/main/annotated_papers/batchNorm_paper.pdf)
- [Language Models are Few-Shot Learners](https://github.com/AngelaYu-3/makemore/blob/main/annotated_papers/openAI_LLM.pdf)

___

## Included Dataset

The included example **shakespeareData.txt** dataset has Shakespeare passages [dataset]([https://www.ssa.gov/](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt)).

___

## License
MIT
