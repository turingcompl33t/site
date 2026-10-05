+++
title = 'TODO'
date = 2026-09-20
slug = 'makemore-batchnorm'
description = 'TODO'
tags = ['ai', 'language-models', 'neural-networks']
+++

In the [prior post]({{< relref "2026-08-22-makemore-mlp" >}}) in this series, we TODO

### NOTES

Karpathy calls backpropagation a [leaky abstraction](https://karpathy.medium.com/yes-you-should-understand-backprop-e2f06eab496b)

you can't just stack up a bunch of differentiable functions, backpropagate through them, and cross your fingers

need to understand how the process works to be able to debug things and ensure we dont introduce new subtle bugs

we already implemented micrograd where we implemented backpropagation with scalars