# CopyNet

CopyNet extends the functionality of encoder-decoder models to allow the generation
of output sequences that contain "out of vocabulary" tokens that appeared in the input sequence.

## About this implementation

This is a re-implementation of [CopyNet](https://arxiv.org/abs/1603.06393), with code based from [code by adamklec](https://github.com/adamklec/copynet).

The differences between this implementation and the previous implementation are:
- a rewritten dataset class that is decoupled from file format
- capability of using a pretrained word embedding (GLOVE in our test case)

The first is the input sequence, the second is the target output sequnce.

The tokens in each sequence should be seperated by spaces.
I used spacy to tokenize the training data so the SequencePairDataset class as well as the evaluation methods assume that spacy will be used.
If you want to use a different tokenizer be sure to update those files accordingly.

Train the model using the train.py script. Most hyperparameters can be tuned with command line arguments documented in the training script.
