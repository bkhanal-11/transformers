# Attention is All You Need

The implementation of transformer as presented in the paper !["Attention is all you need"](https://arxiv.org/abs/1706.03762) from scratch.

Excellent Illustration of Transformers: ![Illustrated Guide to Transformers Neural Network: A step by step explanation](https://www.youtube.com/watch?v=4Bdc55j80l8)

Keys, Queries and Values in Attention Mechanism: ![What exactly are keys, queries, and values in attention mechanisms?](https://stats.stackexchange.com/questions/421935/what-exactly-are-keys-queries-and-values-in-attention-mechanisms#424127)

The Transformer architecture is a popular type of neural network used in natural language processing (NLP) tasks, such as machine translation and text classification. It was first introduced in a paper by Vaswani et al. in 2017.

At a high level, the Transformer model consists of an encoder and a decoder, both of which contain a series of identical layers. Each layer has two sub-layers: a self-attention layer and a feedforward layer. The self-attention layer allows the model to attend to different parts of the input sequence, while the feedforward layer applies a non-linear transformation to the output of the self-attention layer.

Now, let's break down the math behind the self-attention layer. Suppose we have an input sequence of length $N$, represented as a matrix $X$, where each row corresponds to a word embedding. We want to compute a new sequence of vectors $Z$, where each vector is a weighted sum of all the input vectors:

$$
Z = XW
$$

However, we want to compute the weights dynamically, based on the similarity between each pair of input vectors. This is where self-attention comes in. We first compute a "query" vector $Q$, a "key" matrix $K$, and a "value" matrix $V$:

$$
Q = XW_q \\
K = XW_k \\
V = XW_v
$$

where $W_q$, $W_k$, and $W_v$ are learned weight matrices. Then, we compute the attention weights as a softmax function of the dot product between $Q$ and $K$:

$$
\text{Attention}(Q, K, V) = \text{softmax}(\frac{Q K^{T}}{\sqrt{d_k}})V
$$

where $d_k$ is the dimensionality of the key vectors. The softmax function ensures that the attention weights sum to $1$, and the scaling factor of $\frac{1}{\sqrt{d_k}}$ helps stabilize the gradients during training.

Finally, we compute the output of the self-attention layer as a weighted sum of the value vectors:

$$
Z = \text{Attention}(Q, K, V) W_o
$$

where $W_o$ is another learned weight matrix. The output of the self-attention layer is then passed through a feedforward layer with a ReLU activation function, and the process is repeated for each layer in the encoder and decoder.

Overall, the Transformer architecture is a powerful tool for NLP tasks, and its self-attention mechanism allows it to model long-range dependencies in the input sequence.
