## A Scalar-Valued Autograd Engine

In my day to day work, I find that I am spending more and more time building things at very high levels of abstraction. This is useful for getting things done and satisfying requirements, but it makes me nervous that I am losing touch with a deeper understanding of the concepts on which these abstractions are built.

To remedy this, I've started following along with Andrej Karpathy's [Neural Networks: Zero to Hero](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ) YouTube playlist. In it, Andrej starts from calculus (backpropagation and automatic differentiation) and builds to the construction of a language model that is analogous to [GPT-2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf).

I find that attempting to explain a concept to others is the most effective way for me to develop a deep understanding, so in addition to watching Andrej's videos, I'll be writing about the projects here.

This is the first post in this series, in which we implement a scalar-valued automatic differentiation engine. The video on which it is based is [here](https://www.youtube.com/watch?v=VMj-3S1tku0&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=1), and Andrej's original source repository is [on GitHub](https://github.com/karpathy/micrograd).

**TODO**

Why do we want to calculate gradient's automatically? Because this allows us to implement backpropagation, which is the key technique for training neural networks.

In most neural networks, the final layer of the computation that it implements is a _loss function_ - a function that summarizes the performance of the network with respect to some input data. The lower the values produced by this loss function, the better the network is performing with respect to the task of interest.

Within a neural network architecture, we have a set of parameters - weights and biases - that we can tune to tweak the network's performance by changing the computation it implements. In backpropagation, our goal is to compute the contribution of each of these tunable parameters to some observed value of the loss function, which subsequently allows us to automatically compute an update to each parameter that tweaks the network in the direction of making the correct prediction on that input.

Repeating the process iteratively for many inputs, guided by a higher-level algorithm like gradient descent, trains the neural network.

A neural network is just a mathematical expression, and a relatively-simple one at that.

