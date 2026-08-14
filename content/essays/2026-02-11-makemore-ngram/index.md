+++
title = 'N-Grams and Other Experiments'
date = 2026-02-11
slug = 'makemore-ngram'
description = 'Generalizing the statistical bigram model to arbitrary n-grams, and using it as a vehicle to explore hyperparameter tuning.'
tags = ['ai', 'language-models']
+++

In this post, we build on the statistical bigram language model we built in a [previous post]({{< relref "2025-12-02-makemore-bigram" >}}), generalizing it to support n-grams (instead of just bigrams) and using it as a vehicle to explore other machine learning concepts, namely hyperparameter tuning.

## Generalizing Bigrams to an N-Gram Language Model

Porting the existing bigram language model to one that supports character n-grams requires refactoring several implementation details. Here, we'll just look at the major ones. For the full details, you can refer to the [implementation](https://github.com/turingcompl33t/makemore-and-friends/blob/master/makemore/src/makemore/ngram_stat.py) of the `StatisticalNGram` class on GitHub.

**Recording N-Gram Counts / Probabilities**

The most complicated part of generalizing the bigram implementation to n-grams is determining how to represent n-gram counts when the value of `n` is dynamic.

In the bigram language model, we use a 2-dimensional tensor to represent bigram counts (and later the bigram probability distribution). The dimension of this tensor for bigrams is fixed at `27x27`, where `27` is the size of our alphabet - 26 lowercase ASCII characters plus the special character `.`.

To support n-grams, the dimensions of this tensor are no longer fixed. The tensor's structure will depend on the value of `N` - the number of characters considered to predict the next character.

There are multiple ways we might represent n-gram counts and probabilities in a tensor format. A naive generalization of the bigram implementation might just add a dimension for each increase in the value of `N`. As in the current implementation, bigrams would be a `27x27` tensor. Trigrams would add a dimension, `27x27x27`, resulting in a tensor with a total of `19,683` elements.

Another way of looking at representing n-gram counts (and later probabilities) is in terms of _conditioning_ and _prediction_. This models the way that we actually think about the problem in practice: for n-grams, we look at the next `n-1` characters in the input string (the _conditioning_) and use this data to predict what the next character will be (the _prediction_). This method of modeling the problem always yields a two-dimensional tensor, where the cardinality of the first dimension grows with the number of possible strings of length `n-1`.

The logic that handles the "real work" of constructing the counts matrix is in the `_build_lut()` method of the `StatisticalNGram` class. Calculating the unique characters in the alphabet and the alphabet and reverse-alphabet mappings are straightforward - identical to the bigram implementation. To construct the "string-to-index" mapping, we use the following code:

```python
self.stoi = {
    "".join(c): i
    for i, c in enumerate(
        itertools.product(*(chars for _ in range(self.n - 1)))
    )
}
```

The code uses `itertools.product()` to generate all possible sequences of characters (in our alphabet) with length `n-1`. With each of these strings, we assign them a unique index. The "index-to-string" mapping is then just the reverse of this mapping.

With the `stoi` data structure, we initial the counts tensor with:

```python
# initialize counts to 0
N = torch.zeros((len(self.stoi), len(self.alphabet)), dtype=torch.int32)
```

The n-gram probability matrix `P` has the same shape as `N`; it is computed from `N` in a manner identical to that employed for bigrams.

**Extracting N-Grams**

The logic for extracting n-grams from the word list must also change to account for a dynamic value of `n`. This functionality is handled by the `_extract_ngrams()` method in the implementation:

```python
def _extract_ngrams(
    n: int, words: list[str]
) -> Generator[tuple[str, str], None, None]:
    """Extract ngrams from the provided sequence of words."""
    for word in words:
        yield from _extract_ngrams_one(n, word)
```

The method merely iterates over all of the words in the corpus, `yield`ing from `Generator`s produced by the `_extract_ngrams_one()` function, which handles extracting character n-grams from a single word:

```python
def _extract_ngrams_one(
    n: int, word: str
) -> Generator[tuple[str, str], None, None]:
    """Extract ngrams from a single word."""
    chs = [_TOKEN_DOT] * (n - 1) + list(word) + [_TOKEN_DOT] * (n - 1)
    for ngram in zip(*(chs[i:] for i in range(n))):
        yield "".join(ngram[:-1]), ngram[-1]
```

We now pad the word with a variable number of special tokens (`.`); we need `n-1` special tokens at the beginning of the string to handle the initial `n-1` characters of conditioning. We then use `zip` combined with a list-comprehension to generate n-grams by `zip`ing together successfully shorter substrings from the input string. The complete conditioning and prediction are returned as a `tuple`.

## Performance of Bigrams versus Trigrams

With the generalization of the n-gram model in place, it becomes a simple matter to instantiate the n-gram model for both bigrams (`n = 2`) and trigrams (`n = 3`) and evaluate their performance.

```python
from makemore.ngram_stat import StatisticalNGram

model = StatisticalNGram(n=2)
model.train(words)

loss = model.loss(words)
loss
# 2.4540144946949742
```

```python
from makemore.ngram_stat import StatisticalNGram

model = StatisticalNGram(n=3)
model.train(words)

loss = model.loss(words)
loss
# 1.9166365403120391
```

The difference does not appear large in absolute magnitude, but the trigram model results in loss that is about `22%` lower than that produced by the bigram model.

## Hyperparameter Tuning

Hyperparameters are non-learned parameters that control some aspect of an algorithm's learning process. In our implementation, we support one hyperparameter: the degree of model smoothing that is applied. Hyperparameter tuning, then, is the process by which we determine the best setting of hyperparameters through experimentation. In this section, we walk through a simple implementation of hyperparameter tuning for the statistical n-gram model.

The process for hyperparameter tuning looks something like:

1. Split the dataset into training and test datasets
2. Define a hyperparameter search space
3. For each hyperparameter configuration in the search space, train a model (on a subset of the training dataset) and evaluate its performance (on a distinct subset); return the best hyperparameter configuration
4. With the best hyperparameter configuration, train the model on the complete training set and evaluate it on the test set

**Creating a Train / Test Split**

The first utility we need to support hyperparameter tuning is therefore a means of splitting a dataset into `train` and `test` splits. Libraries like [scikit learn](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) can perform this operation for us; below is a simple implementation of this functionality that is specific to our dataset of words:

```python
def train_test_split(data: list[str], test_size: float = 0.2, random_state: int = 0) -> tuple[list[str], list[str]]:
    """Perform a train / test split of the input data."""
    if not 0.0 < test_size < 1.0:
        raise ValueError("test size must be on (0.0, 1.0)")

    random.seed(random_state)

    train, test = [], []
    for element in data:
        if random.random() < test_size:
            test.append(element)
        else:
            train.append(element)

    return train, test
```

We can validate that our `test_size` parameter was respected by computing the relative size of the test set:

```python
train, test = train_test_split(words)
len(test) / len(words)
# 0.19898229950363688
```

**Defining the Search Space**

There are [many ways](https://scikit-learn.org/stable/modules/grid_search.html) of generating a search space for hyperparameter tuning. A simple, if computationally expensive, algorithm is _grid search_ wherein we compute the product of all hyperparameter variables and the values they can take. The code below computes a hyperparameter grid for grid search:

```python
from typing import Any
import itertools

def search_grid(hp: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """Generate the search grid for a given set of hyperparameters."""
    product = itertools.product(*([(k, v) for v in hp[k]] for k in hp.keys()))
    return [
        {k: v for k, v in candidate}
        for candidate in product
    ]
```

To use it, we define the hyperparameter variables in which we are interested, along with the values that they can take:

```python
hyperparameters = {
    "a": [1],
    "b": ["hello", "world"],
    "c": ["foo", "bar", "baz"]
}
```

Invoking our function on this example input produces the following output:

```python
search_grid(hyperparameters)
[{'a': 1, 'b': 'hello', 'c': 'foo'},
 {'a': 1, 'b': 'hello', 'c': 'bar'},
 {'a': 1, 'b': 'hello', 'c': 'baz'},
 {'a': 1, 'b': 'world', 'c': 'foo'},
 {'a': 1, 'b': 'world', 'c': 'bar'},
 {'a': 1, 'b': 'world', 'c': 'baz'}]
```

**Evaluating Performance**

The question that remains is: how do we evaluate the performance of our model with a specific combination of hyperparameters? We could simply rely on our train / test split method from before, but this would imply that for any given hyperparameter configuration, our model is only trained and evaluated once, and therefore we don't get a good sense for how this configuration performs when it is exposed to all of the available data.

K-fold cross validation addresses this problem. In K-fold cross validation, we split the training data into `k` equal-sized subsets (folds). We then train the model `k` times, each time training with `k - 1` folds and evaluating on the remaining fold. In this way, we evaluate our model on every data point exactly once, and we achieve a less-biased estimate of model capability.

To implement K-fold cross validation, we need a way of producing the K-folds of data. The function below splits the input data into `k` folds, allocating additional elements to the final fold if the number of elements in `data` is not evenly-divisible by `k`:

```python
def kfold_split(
    data: list[str], k: int = 3, shuffle: bool = False, random_state: int = 0
) -> list[list[str]]:
    """Generate a random k-fold split."""
    if k < 2:
        raise ValueError("k must be at least 2")
    random.seed(random_state)

    # shuffle if desired
    input = random.sample(data, len(data)) if shuffle else data

    # split and return; the remainder is always allocated to final split
    div, mod = divmod(len(data), k)
    return [input[i * div : (i + 1) * div] for i in range(k - 1)] + [
        input[div * (k - 1) : div * k + mod]
    ]
```

This handling of non-evenly-divisible data is not a perfect solution because it does not evenly allocate data amongst the folds optimally, but it does a job for this implementation.

We can execute the `kfold_split()` function on some dummy data to get a feel for how it operates:

```python
d = [str(i) for i in range(9)]
kfold_split(d)
# [['0', '1', '2'], ['3', '4', '5'], ['6', '7', '8']]
```

With a number of elements that are not evenly divisible, we get an unbalanced fold:

```python
d = [str(i) for i in range(11)]
kfold_split(d)
# [['0', '1', '2'], ['3', '4', '5'], ['6', '7', '8', '9', '10']]
```

Now that we can produce a K-fold split of data, we can wrap this in some additional logic to perform the grouping of the splits for cross validation:

```python
from typing import Generator

def kfold_split_cv(
    data: list[str], k: int = 3, shuffle: bool = False, random_state: int = 0
) -> Generator[tuple[list[str], list[str]], None, None]:
    """Perform k-fold split of input data and return groupings for cross-validation."""
    splits = kfold_split(data, k, shuffle, random_state)
    for i in range(k):
        yield [
            element
            for j, split in enumerate(splits)
            for element in split
            if j != i
        ], splits[i]
```

The `kfold_split_cv()` function just generates all of the appropriate splits for cross-validation, joining the `k - 1` folds into a single data structure for use as training data for the model.

We can again invoke this function on some dummy data to see the splits it produces:

```python
data = [str(i + 1) for i in range(9)]
print([_ for _ in kfold_split_cv(data)])

[(['4', '5', '6', '7', '8', '9'], ['1', '2', '3']),
 (['1', '2', '3', '7', '8', '9'], ['4', '5', '6']),
 (['1', '2', '3', '4', '5', '6'], ['7', '8', '9'])]
```

**Putting it All Together**

Now we put all of this together to perform hyperparameter tuning for our statistical n-gram model. First, we create a global `train` / `test` split of the entire dataset:

```bash
train, test = train_test_split(words)
```

As mentioned above, the only hyperparameter we expose on our `StatisticalBigram` and `StatisticalNGram` models is the `smoothing` parameter which controls the number of "fake counts" that are injected to smooth the probability distribution. We can specify some reasonable values for this hyperparameter:

```python
hyperparameters = {
    "smoothing": [1, 3, 5, 10, 20]
}
```

Now, for various n-gram sizes, we can tune hyperparameters and observe how model smoothing impacts performance:

```python
ngram_size = 2

# compute best hyperparameters via search
best_hp, best_cv_loss = tune_hyperparameters(ngram_size, train, hyperparameters)

# train and evaluate with full sets
model = StatisticalNGram(ngram_size, **best_hp)
model.train(train)

test_loss = model.loss(test)

print(f"{best_hp=}")
print(f"{best_cv_loss=}")
print(f"{test_loss=}")
```

The table below summarizes the results of the experiment:

| NGram Size | Best Hyperparameters | Best CV Loss | Test Loss |
| --- | --- | --- | --- |
| 2 | `{"smoothing": 1}` | `2.46` | `2.46` |
| 3 | `{"smoothing": 1}` | `1.98` | `1.97` |

For both models, optimal performance is achieved when the `smoothing` hyperparameter is set to `1`. This is not surprising. A value of `1` ensures that we never encounter a `0`-count for a particular n-gram, and at the same time does not introduce additional smoothing magnitude that confounds the real counts learned from the dataset.
