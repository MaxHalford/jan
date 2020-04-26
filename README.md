# jan: Just Another Neural network

<div align="center">
    <img src="https://tvguide1.cbsistatic.com/i/2006/10/12/64bce89f-e288-40ff-a717-aaddc756b219/1b7ddbc27ecd03f8ce2443470fc086f3/1CD69017-F556-43A1-866E-0D14FB7A4CC0.jpg">
</div>

This is a very plain neural network library written in Python. It has nothing fancy going on: no automatic differentiation, no GPU support, etc. It has no ambition whatsoever, apart from being used for my own purposes as an educational tool. Therefore, more emphasis is put on readability than on performance.

## Installation

You can install this as a package with `pip`:

```sh
$ pip install git+https://github.com/MaxHalford/jan
```

If you're a student and want to work on the code, here are the instructions to follow:

```sh
# Download the code from GitHub
$ git clone https://github.com/MaxHalford/jan
$ cd path/to/jan

# Create a virtual environment
$ python3 -m venv .
$ source ./bin/activate

# Install in development mode
$ pip install -e ".[dev]"
$ python3 setup.py develop
```

You may now edit the code.

## Examples

You can run the examples to make sure they work by running `pytest`.

## Boston

```py
>>> import numpy as np
>>> from sklearn import datasets
>>> from sklearn import metrics
>>> from sklearn import model_selection
>>> from sklearn import preprocessing

>>> import jan

>>> np.random.seed(42)

>>> X, y = datasets.load_boston(return_X_y=True)
>>> X = preprocessing.scale(X)

>>> X_train, X_test, y_train, y_test = model_selection.train_test_split(
...     X, y,
...     test_size=.3,
...     shuffle=True
... )

>>> nn = jan.NN(
...    dims=(13, 20, 1),
...    activations=(jan.activations.ReLU, jan.activations.Identity),
...    loss=jan.losses.MSE,
...    optimizer=jan.optim.SGD(lr=1e-3)
... )

>>> jan.train(
...    nn=nn, X=X_train, y=y_train,
...    epochs=30, batch_size=8,
...    eval_metric=metrics.mean_absolute_error, print_every=10
... )
[10] train loss: 2.980601
[20] train loss: 2.318972
[30] train loss: 1.997199

>>> y_pred = nn.predict(X_test)
>>> print(metrics.mean_absolute_error(y_test, y_pred))
2.063147

```

## Digits

```py
>>> import numpy as np
>>> from sklearn import datasets
>>> from sklearn import metrics
>>> from sklearn import model_selection
>>> from sklearn import preprocessing

>>> import jan

>>> np.random.seed(42)

>>> X, y = datasets.load_digits(return_X_y=True)

>>> y = np.eye(10)[y]  # one-hot encoding

>>> X_train, X_test, y_train, y_test = model_selection.train_test_split(
...     X, y,
...     test_size=.3,
...     shuffle=True
... )

>>> nn = jan.NN(
...     dims=(64, 15, 10),
...     activations=(jan.activations.ReLU, jan.activations.Sigmoid),
...     loss=jan.losses.MSE,
...     optimizer=jan.optim.SGD(lr=1e-3)
... )

>>> jan.train(
...    nn=nn, X=X_train, y=y_train,
...    epochs=30, batch_size=8,
...    eval_metric=metrics.log_loss, print_every=10
... )
[10] train loss: 0.229828
[20] train loss: 0.137580
[30] train loss: 0.114242

>>> y_pred = nn.predict(X_test)
>>> print(metrics.classification_report(y_test.argmax(1), y_pred.argmax(1)))
              precision    recall  f1-score   support
<BLANKLINE>
           0       0.96      0.98      0.97        53
           1       0.90      0.90      0.90        50
           2       0.96      0.96      0.96        47
           3       0.96      0.98      0.97        54
           4       0.97      0.98      0.98        60
           5       0.98      0.94      0.96        66
           6       1.00      0.96      0.98        53
           7       0.95      0.98      0.96        55
           8       0.89      0.93      0.91        43
           9       0.98      0.95      0.97        59
<BLANKLINE>
    accuracy                           0.96       540
   macro avg       0.96      0.96      0.96       540
weighted avg       0.96      0.96      0.96       540

```
