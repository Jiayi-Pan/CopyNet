# CopyNet

CopyNet extends the functionality of encoder-decoder models to allow the generation
of output sequences that contain "out of vocabulary" tokens that appeared in the input sequence.

## About this implementation

This is a re-implementation of [CopyNet](https://arxiv.org/abs/1603.06393), derived from [code by adamklec](https://github.com/adamklec/copynet).

The differences between this implementation and the previous one are:
- a rewritten dataset class that is decoupled from file format, which makes loading costum data much easier
- capability of using pretrained word embeddings (GLOVE in our test case)

The first is the input sequence, the second is the target output sequnce.

The tokens in each sequence should be seperated by spaces.

## Run
To run the experiment, follow ```playground.ipynb``` step by step.
