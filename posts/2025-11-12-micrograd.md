## A Scalar-Valued Autograd Engine

In my day to day work, I find that I am spending more and more time building things at very high levels of abstraction. This is useful for getting things done and satisfying requirements, but it makes me nervous that I am losing touch with a deeper understanding of the concepts on which these abstractions are built.

To remedy this, I've started following along with Andrej Karpathy's [Neural Networks: Zero to Hero](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ) YouTube playlist. In it, Andrej starts from calculus (backpropagation and automatic differentiation) and builds to the construction of a language model that is analogous to [GPT-2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf).

I find that attempting to explain a concept to others is the most effective way for me to develop a deep understanding, so in addition to watching Andrej's videos, I'll be writing about the projects here.

This is the first post in this series, in which we implement a scalar-valued automatic differentiation engine. The video on which it is based is [here](https://www.youtube.com/watch?v=VMj-3S1tku0&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=1), and Andrej's original source repository is [on GitHub](https://github.com/karpathy/micrograd).

### A Neural Network

Why do we want to calculate gradient's automatically? Because this allows us to implement backpropagation, which is the key technique for training neural networks.

To see this, we'll skip ahead to the end end of this post and look at the mathematical model of a neuron that is used in modern neural networks.

![neuron](https://blogs.cornell.edu/info2040/files/2015/09/VqOpE-1c4xc4y.jpg)

In most neural networks, the final layer of the computation that it implements is a _loss function_ - a function that summarizes the performance of the network with respect to some input data. The lower the values produced by this loss function, the better the network is performing with respect to the task of interest.

Within a neural network architecture, we have a set of parameters - weights and biases - that we can tune to tweak the network's performance by changing the computation it implements. In backpropagation, our goal is to compute the contribution of each of these tunable parameters to some observed value of the loss function, which subsequently allows us to automatically compute an update to each parameter that tweaks the network in the direction of making the correct prediction on that input.

Repeating the process iteratively for many inputs, guided by a higher-level algorithm like gradient descent, trains the neural network.

So, a neural network is just a mathematical expression, and a relatively-simple one at that.

### Derivative of a Simple Function

So, all we need to train a neural network is an efficient way to compute gradients through an arbitrary computation graph (a mathematical expression). We'll build up the intuition for how we can accomplish this step by step, starting with computing the derivative of a simple function:

> $$f(x) = 3x^2 - 4x + 5$$

We can implement this function in Python as:

```python
def f(x) -> float:
    return 3*x**2 - 4*x + 5
```

Then, we visualize the function's behavior by generating some input values and computing the corresponding outputs:

```python
xs = np.arange(-5, 5, 0.25)
ys = f(xs)

plt.plot(xs, ys)
```

The resulting plot is a simple parabola:

TODO: image

With some basic knowledge of calculus, we can compute the derivative of this function symbolically:

> $$f'(x) = 6x - 4$$

Its a simple matter to implement the derivative as a function as well:

```python
def df(x) -> float:
    return 6*x - 4
```

In addition to the symbolic derivative, however, we can use the [definition of the derivative](https://en.wikipedia.org/wiki/Derivative) to compute it numerically. The definition states:

> $$L = \lim_{h -> 0}\frac{f(x + h) - f(x)}{h}$$

Effectively, it says:

> When we change the input by a tiny amount, what is the slope (direction and magnitude) of the response of the output?

We can write a function that evaluates the derivative numerically for the function `f` at a provided input, for a given value of `h`:

```python
def numerical(x, h: float = 0.001) -> float:
    return (f(x + h) - f(x)) / h
```

The quality of the estimate provided by the numerical derivative is controlled by the magnitude of the "nudge", `h`. We can look at how much the numerical derivative differs from the expected result provided by the symbolic derivative as a function of `h`:

```python
for h in [0.1, 0.001, 0.0001, 0.00001, 0.000001]:
    print(abs(numerical(4, h=h) - df(4)))
```

Here, the results look like:

```
0.2999999999999403
0.0030000000026575435
0.00029999997821050783
2.9998357149452204e-05
2.990627763210796e-06
```

So, for sufficiently small values of `h`, the numerical derivative provides a close approximation of the symbolic derivative. We'll use this fact to check our work in the coming section when we begin work on extending this reasoning to operate on more complicated functions.

### A Function with Multiple Inputs

We can apply the same logic to functions with multiple inputs. Consider the function:

```python
def f(a, b, c) -> float:
    return a*b + c
```

Now, because there are multiple input variables (`a`, `b`, and `c`) it doesn't make sense to talk about _the_ derivative of the output of this function. Instead, we talk about the derivative of the output with respect to _some_ input. That is, with:

> $$y = a*b + c$$

We have three potential derivatives of interest:

> $$\frac{dy}{da}, \frac{dy}{db}, \frac{dy}{dc}$$

Or, the derivative of `y` with respect to `a`, the derivative of `y` with respect to `b`, and the derivative of `y` with respect to `c`.

We can use the same method as before to approximate these derivatives numerically:

```python
h = 0.0001

a = 2.0
b = -3.0
c = 10.0

# dy/da
(f(a + h, b, c) - f(a, b, c)) / h # approx -3.0
# dy/db
(f(a, b + h, c) - f(a, b, c)) / h # approx 2.0
# dy/dc
(f(a, b, c + h) - f(a, b, c)) / h # approx 1.0
```

These values match up with what we would find by evaluating this derivative symbolically.

### The `Value` Object

Now that we have some intuition about what the derivative is telling us about these expressions, we can start working through the process of generalizing the ability to compute derivatives through an arbitrary computation graph.

As these expressions grow larger, or arbitrarily large, we need an automated way to maintain the computation graph that the expression represents. This way, we can construct a potentially-complicated expression and have the computation graph maintained for us automatically, which will make the task of computing gradients through the graph much simpler.

The `Value` object is the data structure that we'll use to maintain this computation graph. Each instance of `Value` represents a simple, floating-point numeric value, but it has the additional responsibility to maintain some metadata like:

- The input values that were used to produce it, if any
- The operation that was used to combine these input values to produce it, if any
- The gradient of the output of the expression with respect to this particular node in the computation graph

The basic setup for the `Value` object looks like:

```python
from __future__ import annotations

class Value:
    def __init__(self, data: float, _children: tuple[Value,...]=(), _op="", label=""):
        # the data maintained by this object
        self.data = data
        # the gradient of the output of the graph w.r.t this node
        self.grad = 0.0
        # a human-readable label for this node
        self.label = label

        # the function for computing the local gradient
        self._backward = lambda: None
        # the anscestors of this node in the graph
        self._prev = set(_children)
        # the operation used to compute this node
        self._op = _op
```

We can instantiate a `Value` object with some data:

```python
a = Value(a, label="a")
a
# Value(data=2.0)
```

Aside from this, though, we can't really do anything with it yet. For instance, we can't add two `Value`s together:

```python
a = Value(1.0, label="a")
b = Value(2.0, label="b")
a + b
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
Cell In[24], line 3
      1 a = Value(1.0, label="a")
      2 b = Value(2.0, label="b")
----> 3 a + b

TypeError: unsupported operand type(s) for +: 'Value' and 'Value'
```

To start working with these objects to build an expression, we'll need to implement some operations for them. We can start by implementing the `__add__` method to support addition:

```python
def __add__(self, input: float | int | Value) -> Value:
    # wrap other in a Value if not already
    other = input if isinstance(input, Value) else Value(input)

    out = Value(self.data + other.data, (self, other), "+", _next_char((self.label, other.label)))
    return out
```

This method supports addition with both numeric literals as well as other `Value` object:

```python
a = Value(1.0, label="a")
b = Value(2.0, label="b")
c = a + b
# Value(data=3.0)

d = a + 3.0
# Value(data=4.0)
```

We can also implement the `__radd__` function to add support for addition on the right with literals:

```python
def __radd__(self, other: float | int | Value) -> Value:
    return self + other
```

Now, expressions like the following are also supported:

```python
a = Value(1.0, label="a")
d = 3.0 + a
# Value(data=4.0)
```

In addition to addition, multiplication is another fundamental operation that we'll need:

```python
def __mul__(self, input: float | int | Value) -> Value:
    other = input if isinstance(input, Value) else Value(input)

    out = Value(self.data * other.data, (self, other), "*", _next_char((self.label, other.label)))
    return out

def __rmul__(self, input: float | int | Value) -> Value:
    return self * input
```

The implementation is analogous to the implementation of addition. Now, we can construct expressions like the one we used as an example in the [previous section](#a-function-with-multiple-inputs):

```python
a = Value(2.0, label="a")
b = Value(-3.0, label="b")
c = Value(10.0, label="c")

y = a*b + c
y
# Value(data=4.0)
```

More important than just computing the result of the expression (equivalent to the _forward pass_, as we'll come to see) is the fact that the `Value` instance `y` "knows" the complete computation graph that is used to compute it.

A convenient way of demonstrating this is a visual depiction of the computation graph. We won't go into the details for it here, but the repository provides [some code](https://github.com/turingcompl33t/makemore-and-friends/blob/master/micrograd/src/micrograd/util/draw.py) (copied verbatim from [Andrej's implementation](https://www.youtube.com/watch?v=VMj-3S1tku0&t=296s)) that is capable of displaying this graph for us:

```python
from micrograd.util.draw import draw_dot
draw_dot(y)
```

TODO: image.

### Manual Backpropagation

So far, we can build scalar-valued mathematical expressions (limited to addition and multiplication currently) and compute a "forward pass," during which the computation graph for the expression is generated automatically by our `Value` objects. Next, we need to start working on computing gradients throughout this graph via backpropagation.

In this section, we'll work through the backpropagation process manually for an expression to get a sense for how the calculations work. Once that is done, we'll have all of the knowledge we need to automate the calculation of gradients through the entire graph.

We'll use the slightly-more complex expression as our testbed for this exercise:

```python
a = Value(2.0, label="a")
b = Value(-3.0, label="b")
c = Value(10.0, label="c")
e = a*b; e.label="e"
d = e + c; d.label="d"
f = Value(-2.0, label="f")
L = d*f; L.label="L"
```

The choice of the variable name `L` here is deliberate - we can think of this as representing the loss function for a neural network, and by performing backpropagation we'll be computing the contribution of each node to the output value for the loss. If we were training a neural network, this would in turn tell us how to adjust the node values (assuming they are parameters) to reduce the loss and increase the network's performance.

We can visualize the computation graph for this expression with the `draw_dot` function:

TODO: insert.

As we can see in the graph visualization, all of the gradients are currently `0.0`. Our job is to fill in each of these values - we need to compute the derivative of the loss, `L`, with respect to each of `a`, `b`, `c`, etc.

At first glance, this process appears daunting. For instance, if we were begin by considering our variables left to right, how would we compute the derivative:

> $$\frac{dL}{da}$$

Calculus' [chain rule](https://en.wikipedia.org/wiki/Chain_rule) tells us that, generally, if a variable _z_ depends on a variable _y_, which in turn depends on _x_, then _z_ depends on _x_ as well, via the intermediate variable _y_, and that the relationship between them is:

> $$\frac{dz}{dx} = \frac{dz}{dy} * \frac{dy}{dx}$$

Applying this same logic to our expression, we have:

> $$\frac{dL}{da} = \frac{dL}{de} * \frac{de}{da}$$

So, we can compute the derivative of the loss, `L`, with respect to `a` given the derivative of `L` with respect to the intermediate variable `e` and the local derivative $\frac{de}{da}$.

The situation is analogous for every other node in the graph, and this fact reveals a critical insight. In backpropagation, we start at the output and work backwards to compute the gradient of the output with respect to each node in the graph. We work backwards because, according to chain rule, computing the gradient for a given node is straightforward _if_ we already know the gradient with respect to intermediate variable that is nearest to the target variable in the graph (and on the path to the output variable).
