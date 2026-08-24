+++
title = ''
date = 2026-08-20
slug = 'makemore-mlp'
description = 'Extending makemore with an MLP, and running some experiments to minimize validation loss'
tags = ['ai', 'language-models', 'neural-networks']
draft = true
+++

In a [prior post]({{< relref "2026-08-15-makemore-nn" >}}) in this series, we built a simple character-level language model with a single-layer neural network and optimized it via gradient descent. In this post, we'll extend the existing neural network architecture with multiple layers, and run some more in-depth experiments to tune our model's hyperparameters and minimize validation loss.

As before, the content in this post is based on Andrej Karpathy's [YouTube video](https://www.youtube.com/watch?v=TCH_1BHY58I&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=3) wherein he describes all of these concepts. The model itself is based on a 2003 implementation by [Bengio et al.](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf). All of the source code that I produced for this post is available in my [`makemore-and-friends`](https://github.com/turingcompl33t/makemore-and-friends) repository. 

### Weakness of the Prior Approach

Weaknesses of the previous model: predictions are not very good. this is a result of the fact that we only take 1 previous character into account.

if we take just one prior character as context, we have a counts matrix that is 27 rows (one for each character).
if we take two prior characters, we have 27x27 rows which is 729 rows.
with three, it is 27**3 which is 19683 rows

the number of rows in this matrix grows exponentially in the number of characters in the context. the approach is too naive to scale well.

### The Updated Approach

We build a character level model where they build a word level model; the modeling approach is the same though.

they have a vocabulary of 17,000 words that they embed into a 30-dimensional feature space. this is a massive reduction in dimensionality of the input space.

we will tune these embeddings via gradient descent.

we can perform the same embedding process for our characters; taking a 27-dimensional input space to one that is much smaller.

use a multi-layer neural network to predict the next character; optimize it by minimizing negative log likelihood. 

**Intuition**

![intuition](./intuition.png)

the intuition (insert here) - we can use the embedding space to transfer knowledge

**Architecture**

![architecture](./architecture.png)

take three previous words as context and try to predict the next word

lookup table `C` that gets the embedding for each word in the vocabulary. therefore, the lookup table has shape `(17,000, 30)`, assuming that 30 is the embedding dimension.

the input layer is 30 neurons for 3 words = 90 neurons total in the input layer.

the size of the hidden layer is a hyperparameter of the network. this layer is fully connected to the input layer. `tanh` nonlinearity.

the output layer has 17,000 neurons - one for each word in the vocabulary. softmax layer to get a probability distribution over the next word in the sequence.

### Building our Dataset

The dataset is a just a file with 32,033 names, one per line:

```python
words = load_names(data_dir / "names.txt")
print(len(words))  # 32033
print(words[:4])   # ['emma', 'olivia', 'ava', 'isabella']
```

In the `makemore` library, I wrote a helper function that builds the dataset vocabulary from the input words. Invoking it looks like:

```python
vocab = Vocab.from_words(words)
```

Under the hood, it just builds the lookup tables from index to string and string to index. It also defines `.` as the special boundary token, for a total vocabulary size of 27 characters:

```python
class Vocab:
    # ...

    # index-to-string
    self.itos: list[str] = [TOKEN_BOUNDARY, *sorted(chars)]
    # string-to-index
    self.stoi: dict[str, int] = {c: i for i, c in enumerate(self.itos)}
```

Now, to actually make our dataset, we need to split up the input examples into _context_ and _prediction_ - context are the `n` prior characters that we want to consider when predicting the next character, and prediction is the character that is being predicted. The parameter `block_size` controls the size of the context. 

```python
X, Y = make_dataset(words, vocab, block_size=3)
```

```python
xs: list[list[int]] = []
ys: list[int] = []

for word in words:
    context = [vocab.stoi[TOKEN_BOUNDARY]] * block_size
    for ch in list(word) + [TOKEN_BOUNDARY]:
        ix = vocab.stoi[ch]
        xs.append(context)
        ys.append(ix)
        # roll the context forward by one token
        context = context[1:] + [ix]

return torch.tensor(xs), torch.tensor(ys)
```

Now we can run this code to produce datasets for various block sizes. Initially we just run this with the first 5 words in the corpus to make things easier. This results in 32 examples for training while we build the network.

when `block_size=2`:

```
.. --> e
.e --> m
em --> m
mm --> a
ma --> .
```

when `block_size=3`:

```
... --> e
..e --> m
.em --> m
emm --> a
mma --> .
```

when `block_size=4`:

```
.... --> e
...e --> m
..em --> m
.emm --> a
emma --> .
```

notice that the number of examples produced by each word in the input does not change with the block size--the word `emma` always results in five examples for training, regardless of what the block size is.

```python
print(X.shape, X.dtype, Y.shape, Y.dtype)
# (torch.Size([32, 3]), torch.int64, torch.Size([32]), torch.int64)
```

### Building the Network

Start by building the lookup table `C`. Initialize it randomly:

```python
C = torch.randn((len(vocab), embedding_dimension))
```

There is equivalence between indexing with the index that corresponds to the character of interest, and matrix multiplication with the one-hot encoding of our vocabulary.

```python
C[vocab.stoi["e"]]
# tensor([-0.4712, -1.3342])
```

```python
F.one_hot(torch.tensor(vocab.stoi["e"]), num_classes=len(vocab)).float() @ C
# tensor([-0.4712, -1.3342])
```

We can interpret this first embedding layer as a means of selecting the embedding for each character in our context.

**Embedding All Inputs Simultaneously**

We can index into a PyTorch `tensor` with other tensors, including multidimensional ones.

For each row in `X`, we want to get the embedding from `C` for each of `block_size` (3) input characters in the context.

If we look at a single row of `X`, we have three indices, corresponding to the three characters in the block:

```python
X[0]
# tensor([0, 0, 0])
```

If we index into `C` with this tensor, we get the corresponding embedding for each of the three characters:

```python
C[torch.tensor([0,0,0])]
# tensor([[ 0.9462, -0.0975],
#        [ 0.9462, -0.0975],
#        [ 0.9462, -0.0975]])
```

But we can just index directly into the tensor with `X`:

```python
C[X].shape
# (32, 3, 2)
```

For `C` of shape `(V, D)` and an integer tensor `idx` of any shape `S`:

```
C[idx].shape == S + C.shape[1:]
```

Here we have `C` with shape `(27, 2)` and integer tensor `idx` with shape `(32, 3)`. The property holds:

```
# S.shape   # C.shape[1:]    # C[idx].shape
(32, 3)   + (,2)          == (32, 3, 2)
```

The indexed dimension (`V`) is replaced by the entire shape of the index tensor, and the remaining dimensions of  `C` are appended unchanged.

**Constructing the Hidden Layer**

The number of inputs to this layer varies with two things:

- the `block_size`
- the embedding dimension

Together, these two determine the number of neurons on the previous (input) layer.

need to transform the `(32, 3, 2)` output from the previous layer to a `(32, 6)` in order to perform naive matrix multiplication here to implement the forward pass

the simplest way to do this is to use [`torch.view`](https://docs.pytorch.org/docs/2.13/generated/torch.Tensor.view.html). this is efficient because it doesn't allocate any new storage; it merely changes some metadata about the tensor.

the way our `emb` tensor gets flattened to make it `(32, 6)` is the desired way.

we apply the [`tanh`](https://docs.pytorch.org/docs/2.13/generated/torch.tanh.html) activation function as well.

```python
h = torch.tanh(emb.view((-1, 6)) @ W1 + b1)
h.shape # (32, 100)
```

**The Final Layer and Loss**

```python
W2 = torch.randn((100, len(vocab)))
b2 = torch.randn(len(vocab))
```

```python
logits = h @ W2 + b2
logits.shape # (32, 27)
```

To compute negative log likelihood, we need to exponentiate the logits and normalize across rows to create a probability distribution. previously, we used a combination of `.exp()` and manual summation / division to implement this.

```python
counts = logits.exp()
probs = counts / counts.sum(axis=1, keepdim=True)
loss = -probs[torch.arange(32), Y].log().mean()
loss # 14.7941
```

the better way is to use [`torch.nn.functional.cross_entropy`](https://docs.pytorch.org/docs/2.13/generated/torch.nn.functional.cross_entropy.html)

```python
import torch.nn.functional as F
loss = F.cross_entropy(logits, Y)
loss # 14.7941
```

I discuss some of the reasons to prefer `F.cross_entropy()` to the hand-rolled implementation in [[1]](#footnotes).

**The Complete Forward Pass**

The complete forward pass looks like this:

```python
# embed all blocks for each input example
emb = C[X]
# hidden layer activations
h = torch.tanh(emb.view((-1, 6)) @ W1 + b1)
# logits (activations from output layer)
logits = h @ W2 + b2
# NLL loss
loss = F.cross_entropy(logits, Y)
```

### Aside: `torch.cross_entropy` vs Hand-Rolled Loss

- this is more efficient during the forward pass because PyTorch can used a fused kernel under the hood instead of implementing all of the intermediate operations
- this is more efficient during the backward pass as well; we can derive a simpler derivative for the analytical form of this operation (show micrograd `tanh`)
- also more stable numerically

### Wrapping the Forward Pass and a Simple Training Loop

We can wrap the complete forward pass in the `.forward()` method on the `MLP` class. 

```python
# class MLP
def forward(self, ctx: torch.Tensor) -> torch.Tensor:
    emb = self.C[ctx]
    # hidden layer activations
    h = torch.tanh(
        emb.view((-1, self.block_size * self.embedding_dimension)) @ self.W1
        + self.b1
    )
    # logits (activations from output layer)
    logits = h @ self.W2 + self.b2
    return logits
```

Now, computing the forward pass and loss looks like:

```python
from makemore.mlp import MLP

model = MLP(vocab)
logits = model.forward(X)
loss = F.cross_entropy(logits, Y)
```

A simple training loop.

```python
for _ in range(10):
    # forward pass
    logits = model.forward(X)
    loss = F.cross_entropy(logits, Y)
    print(loss.item())

    # backward pass
    for p in model.parameters():
        p.grad = None
    loss.backward()

    # update
    for p in model.parameters():
        p.data += -0.1 * p.grad # type: ignore
```

### Finding a Good Initial Learning Rate

TODO

### Beating Andrej's Best Validation Loss

```
found best setting: {'embedding_dimension': 10, 'hidden_layer_size': 500} (n_params=29297)
best validation loss = 2.211148738861084
```

Andrej's best = 2.17

I think there is still further tweaking to be done here in on the following areas:

- Modifying the block size
- Modifying the batch size
- Adjusting the optimization process (number of training steps, learning rate schedule) according to the hyperparameter setting

### References

- [_A Neural Probabilistic Language Model_](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf) - Bengio et al.
- [PyTorch Internals](https://blog.ezyang.com/2019/05/pytorch-internals/) - Eric Yang's blog

### Footnotes

**[1] Preferring `F.cross_entropy()**

There are some distinct, compelling reasons to prefer `F.cross_entropy()` from `torch` to a hand-rolled implementation of negative log likelihood loss. The first is just the amount of code we have to write. Our hand-rolled implementation looked like:

```python
counts = logits.exp()
probs = counts / counts.sum(axis=1, keepdim=True)
loss = -probs[torch.arange(32), Y].log().mean()
```

where `F.cross_entropy()` reduces this to a single line of more-expressive code.

Beyond the code's conciseness and expressiveness, the implementation from `torch` is also more efficient when computing the forward pass and the backward pass.

In our hand-rolled implementation, each individual operation (e.g `exp()`, `.sum()`, division, etc.) is computed individually and materialized as a new tensor. This is needlessly expensive because all of the intermediate results are irrelevant and we throw them away. When we use `F.cross_entropy()`, we express the fact that what we're really interested in is the negative log likelihood loss (not the raw counts, not the probabilities, etc.) and `torch` can execute all of the mathematical operations required to achieve this result more efficiently This is similar to the way that [kernel fusion](https://pytorch.org/blog/why-is-pytorch-compile-so-fast-kernel-fusion/) works when compiling `torch` models.

During the backward pass, we have to compute gradients through the entire computation graph implied by our model. One way we could do this is by calculating derivatives through every operation of the loss computation, but its more efficiently analytically to just compute a single derivative - the one that implements the backward pass through cross-entropy loss. We saw this in practice when implementing the `backward()` operation for the `tanh` method (for example) in [`micrograd`](https://github.com/turingcompl33t/makemore-and-friends/blob/25b7458fb395c1fd52ffe2a8c625edb3737553ed/micrograd/src/micrograd/engine.py#L90). The forward pass calculation looked like:

```python
t = (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))
```

but instead of computing derivatives through each of these individual operations (exponentiation, division, addition, etc.) we just derived the derivative for hyperbolic tangent analytically and implemented this directly:

```python
def _backward():
    self.grad += (1 - t**2) * out.grad
```

This meant that a backward pass through `tanh` required far fewer calculations than it would in the naive implementation, and the same is true here for our loss computation with `F.cross_entropy()`.

Finally, `torch` implements some tricks internally to ensure that the computation is more numerically well-behaved than a naive implementation. Consider a simple example:

```python
logits = torch.tensor([-2, -3, 0, 5])
counts = logits.exp()
probs = counts / counts.sum()
probs
# tensor([9.0466e-04, 3.3281e-04, 6.6846e-03, 9.9208e-01])
```

Here, everything is good--the output is a probability distribution. Now consider what happens when one of the logits takes on an extreme negative value:

```python
logits = torch.tensor([-100, -3, 0, 5])
counts = logits.exp()
probs = counts / counts.sum()
probs
# tensor([0.0000e+00, 3.3311e-04, 6.6906e-03, 9.9298e-01])
```

This is still well-behaved; everything looks like a probability and they sum to 1. However, the same is not true when an extreme positive value is present:

```python
logits = torch.tensor([-100, -3, 0, 100])
counts = logits.exp()
probs = counts / counts.sum()
probs # tensor([0., 0., 0., nan])
```

Now we end up with `nan` in our final probability distribution. The reason is that we are providing a large number (`100`) to the `exp()` function. When we exponentiate with this, we overflow the range of the underlying data type:

```python
counts
# tensor([3.7835e-44, 4.9787e-02, 1.0000e+00,        inf])
```

and we end up with an `inf` in our intermediate `counts` array.

The key to resolving this issue is that we can perform arbitrary addition / subtraction to the input `logits` array and get the same result, because of the normalization that occurs. We don't care about the absolute value of the logits, merely their relationship to one another when we are calculating the probability distribution over them. Therefore, we can subtract the highest magnitude value that occurs in the input tensor from all of the elements of the tensor and achieve the same result:

```python
logits = torch.tensor([-100, -103, 0, 100]) - 100
counts = logits.exp()
probs = counts / counts.sum()
probs
# tensor([0.0000e+00, 0.0000e+00, 3.7835e-44, 1.0000e+00])
```

With the offset, the output distribution is well-behaved again. This is the transformation that `F.cross_entropy()` performs internally to ensure that the result is numerically well-behaved, and it is yet another reason to prefer the implementation from `torch` to our own version.
