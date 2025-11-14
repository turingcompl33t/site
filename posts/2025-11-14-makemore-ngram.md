## N-Grams and Other Experiments

In this post, we build on the statistical bigram language model we built in the previous post, generalizing it to support n-grams (instead of just bigrams) and using it as a vehicle to explore other machine learning concepts, namely hyperparameter tuning.

### Generalizing the Bigram Model

TODO

### Performance of Bigrams versus Trigrams

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

The difference does not appear large in absolute magnitude, but the trigram model results in loss that is about 22% lower than that produced by the bigram model.

### Hyperparameter Tuning

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

```python
train, test = train_test_split(words)
len(test) / len(words)
# 0.19898229950363688
```

Now we need a means of generating the hyperparameter grid for grid search.

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

```python
hyperparameters = {
    "a": [1],
    "b": ["hello", "world"],
    "c": ["foo", "bar", "baz"]
}

search_grid(hyperparameters)
```

Produces the following output:

```python
[{'a': 1, 'b': 'hello', 'c': 'foo'},
 {'a': 1, 'b': 'hello', 'c': 'bar'},
 {'a': 1, 'b': 'hello', 'c': 'baz'},
 {'a': 1, 'b': 'world', 'c': 'foo'},
 {'a': 1, 'b': 'world', 'c': 'bar'},
 {'a': 1, 'b': 'world', 'c': 'baz'}]
```

How do we evaluate the performance of our model with a specific combination of hyperparameters? We accomplish this through K-fold cross validation.

To implement K-fold cross validation, we need a way of producing the K-folds of data 

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

```python
d = [str(i) for i in range(9)]
kfold_split(d)
# [['0', '1', '2'], ['3', '4', '5'], ['6', '7', '8']]
```

It also works for input data that does not divide evenly amongst the splits. It does this by allocating the remainder after integer division to the final fold. This is not a perfect solution because it does not evenly allocate data amongst the folds optimally, but it does a job for this implementation.

```python
d = [str(i) for i in range(11)]
kfold_split(d)
# [['0', '1', '2'], ['3', '4', '5'], ['6', '7', '8', '9', '10']]
```

We wrap this in some additional logic to perform the grouping of the splits for cross validation.

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

```python
data = [str(i + 1) for i in range(9)]
[_ for _ in kfold_split_cv(data)]
```

```python
[(['4', '5', '6', '7', '8', '9'], ['1', '2', '3']),
 (['1', '2', '3', '7', '8', '9'], ['4', '5', '6']),
 (['1', '2', '3', '4', '5', '6'], ['7', '8', '9'])]
```

Now we put all of this together to perform hyperparameter tuning for our statistical ngram models.

```bash
# create our global train / test split
train, test = train_test_split(words)
```

```python
hyperparameters = {
    "smoothing": [1, 3, 5, 10, 20]
}
```

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

The table below summarizes the results of the experiment.

| NGram Size | Best Hyperparameters | Best CV Loss | Test Loss |
| --- | --- | --- | --- |
| 2 | `{"smoothing": 1}` | `2.46` | `2.46` |
| 3 | `{"smoothing": 1}` | `1.98` | `1.97` |
