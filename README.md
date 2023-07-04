# CopyNet: Enhanced Encoder-Decoder Model

CopyNet enhances the capability of standard encoder-decoder models, allowing them to generate output sequences that include "out of vocabulary" tokens present in the input sequence.

## About this Re-implementation

This project is a refined re-implementation of [CopyNet](https://arxiv.org/abs/1603.06393), which originally inspired from [code by adamklec](https://github.com/adamklec/copynet). This version introduces several improvements over the original:

- A completely re-engineered dataset class that is abstracted from file formats, thus making it simpler to load your custom data.
- Integration with pretrained word embeddings (demonstrated using GLOVE), which facilitates improved model performance.

Each line of input is composed of two sequences: the first represents the input sequence, while the second acts as the target output sequence. Tokens in each sequence should be separated by spaces.

## Execution
To utilize this implementation, follow the instructions provided in ```playground.ipynb``` in sequence.
