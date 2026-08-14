+++
title = 'A Neural Character Bigram Language Model'
date = 2026-02-12
slug = 'makemore-nn'
description = 'A neural character-level bigram language model, optimized with gradient descent rather than counted from the training data.'
tags = ['ai', 'language-models', 'neural-networks']
+++

The input to the neural network is a single character.
The output of the neural network is a probability distribution over the predicted next character in the sequence.

We already have a loss function, so we have a means of evaluating the model's prediction. This will allow us to automatically optimize the network's parameters with gradient-based optimization.

## Create the Training Set

Print the bigrams provided by a single word. Show how this is reflected in the tensor representation.

## Inputs

How do we provide this data as input to the neural network? We can use one-hot encoding.

We can use one-hot encoding to encode integers as vectors.

When we are feeding values into a neural network, we want values to be floating point values, not integers.

## Our First Neuron

Get a random vector of weights, randomly initialized. `W` is a column vector with dimensions `(27, 1)`.

This makes the matrix multiplication operation work. 

Also, this operation computes the activation for each of the inputs at the same time. This is what is reflected in the output, another column vector with dimensions `(5, 1)`.

This is just a single neuron.

However, we want 27 neurons, not just a single neuron.

Now, when we do the multiplication `xenc @ W`, we get the activations for each of the 5 inputs, for each of the 27 neurons in the layer. The output is a tensor with shape `(5, 27)`. The first row corresponds to all of the activations for the first input, and so on.

Tensor operations like this make large multiplication operations like this efficient.

## Transforming Outputs

Right now, we just do the operation `x @ W`. There is no bias term. There is no nonlinearity.

This is all we are going to do for this implementation. The simplest possible network.

For each input example, we are trying to produce a probability distribution for the next character in the sequence.

We want something similar to what we had last time. In our counts tensor, for each character, we had a probability distribution for each of the next characters.

This is not what we have right now. We have both negative and positive numbers.

These 27 numbers give us log counts. These are called "logits" I think? Therefore, to get counts we can exponentiate the log counts.

What does the exponentiation function look like?

Now, with these transformations, we can interpret the output of the neural network as next character probabilities. Its also important that all of these operations are differentiable.

Trace a single example through the neural network - a single character input to the probability distribution over next characters.

Now all we have to do is determine if we can find a setting for the parameters in `W` that makes the probability distribution "good" in the sense that it makes loss low.

These last two operations together, exponentiation and normalization are called the softmax.

## A Walkthrough of What the Network Currently Does

For all of the bigrams from the first word in the dataset, look at the bigram, the probability the network currently assigns, and the associated loss (NLL).

also compute the aggregate loss, the average NLL

then show how we can re-sample `W` to change the aggregate loss. This foreshadows how we will improve the model through training.

We can think of random guess and check as a terrible optimization process. We'll implement a better optimization technique.

## Vectorized Loss

Currently, the network outputs a probability distribution for each example. For each example, we want the probability that is assigned to the known correct next character to compute the loss.

## Backward Pass + Update

Walk through the semantics of the gradients of `W` after computing it. This value here indicates how this parameter contributes to the loss.

## Putting it Together

TODO

## Encouraging Smoothness through Regularization

`(W**2).mean()` incurs loss whenever `W` is not zero.

Add this regularization term to the existing loss, with some strength hyperparameter e.g. `0.01`.
