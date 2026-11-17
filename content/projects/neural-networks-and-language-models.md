+++
title = 'Neural Networks and Language Models'
date = 2025-12-02
until = 'present'
description = 'A series building neural networks and language models from scratch, starting at automatic differentiation and working up to models that generate natural language. Each post has accompanying source code.'

[[links]]
  label = 'code'
  url = 'https://github.com/turingcompl33t/makemore-and-friends'
+++

I studied machine learning in pursuit of my MS at Carnegie Mellon University, but not to the extent necessary to (a) understand language models well enough to design novel evaluations or (b) make concrete contributions to technical AI safety.

As a consequence, I started a self-study protocol for ML, with a particular focus on neural networks and language models. This series of mini-projects, and the posts that accompany them, is the result.

The series works through building neural networks and language models from scratch, starting at automatic differentiation and working up to models that generate natural language. Each post has accompanying source code.

Each post is accompanied by source code in the [`makemore-and-friends`](https://github.com/turingcompl33t/makemore-and-friends) repository.

## Neural Networks — Autograd

- [A Scalar-Valued Autograd Engine]({{< relref "2025-11-18-micrograd0" >}})
- [More Autograd: Arithmetic Operations, Nonlinearities, and a Simple Neural Network Library]({{< relref "2025-11-27-micrograd1" >}})

## Language Models

- [A Statistical Character Bigram Language Model]({{< relref "2025-12-02-makemore-bigram" >}})
- [N-Grams and Other Experiments]({{< relref "2026-02-11-makemore-ngram" >}})
- [A Neural Character Bigram Language Model]({{< relref "2026-08-15-makemore-nn" >}})
- [Next Token Prediction with MLP]({{< relref "2026-08-22-makemore-mlp" >}})
