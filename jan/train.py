import numpy as np


def train(nn, X, y, epochs: int, batch_size: int, eval_metric: callable, print_every=np.inf):
    """Trains a neural network.

    Parameters:
        X: array of shape (n_samples, n_features)
        y: array of shape (n_samples, n_targets)
        epochs
        batch_size

    """

    # As a convention, we expect y to be 2D, even if there is only one target to predict
    if y.ndim == 1:
        y = np.expand_dims(y, axis=1)

    # Go through the epochs
    for i in range(epochs):

        # Shuffle the data
        idx = np.arange(X.shape[0])
        np.random.shuffle(idx)
        x_ = X[idx]
        y_ = y[idx]

        # Iterate over the training data in mini-batches
        for j in range(X.shape[0] // batch_size):
            start = j * batch_size
            stop = (j + 1) * batch_size
            nn.partial_fit(x_[start:stop], y_[start:stop])

        # Display the performance every print_every eooch
        if (i + 1) % print_every == 0:
            y_pred = nn.predict(X)
            print(f'[{i+1}] train loss: {eval_metric(y, y_pred)}')
