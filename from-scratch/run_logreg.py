import numpy as np
from logistic_regression import LogisticRegression

def main():
    np.random.seed(42)

    # Create three clusters for three different classes.
    n_samples_per_class = 100

    class_0 = np.random.randn(n_samples_per_class, 2) + np.array([-3, -3])
    class_1 = np.random.randn(n_samples_per_class, 2) + np.array([3, -3])
    class_2 = np.random.randn(n_samples_per_class, 2) + np.array([0, 3])

    X = np.vstack((class_0, class_1, class_2))

    y = np.concatenate((
        np.zeros(n_samples_per_class, dtype=int),
        np.ones(n_samples_per_class, dtype=int),
        np.full(n_samples_per_class, 2, dtype=int)
    ))

    # Shuffle the dataset.
    indices = np.random.permutation(X.shape[0])
    X = X[indices]
    y = y[indices]

    # Split into training and testing sets.
    split_index = int(0.8 * X.shape[0])

    X_train = X[:split_index]
    y_train = y[:split_index]

    X_test = X[split_index:]
    y_test = y[split_index:]

    # Standardize using only training-set statistics.
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)

    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    # Train the model.
    model = LogisticRegression(
        lr=0.1,
        max_iter=10_000,
        tol=1e-10,
        reg="l2"
    )

    model.fit(X_train, y_train, C=3)

    # predict() currently returns probabilities.
    train_probabilities = model.predict(X_train)
    test_probabilities = model.predict(X_test)

    # Convert probabilities into predicted class labels.
    train_predictions = np.argmax(train_probabilities, axis=1)
    test_predictions = np.argmax(test_probabilities, axis=1)

    train_accuracy = np.mean(train_predictions == y_train)
    test_accuracy = np.mean(test_predictions == y_test)

    print(f"Iterations: {len(model.loss_history)}")
    print(f"Initial loss: {model.loss_history[0]:.6f}")
    print(f"Final loss:   {model.loss_history[-1]:.6f}")
    print(f"Training accuracy: {train_accuracy:.2%}")
    print(f"Testing accuracy:  {test_accuracy:.2%}")

    print("\nFirst 10 test predictions:")
    for i in range(min(10, len(y_test))):
        print(
            f"True class: {y_test[i]}, "
            f"Predicted class: {test_predictions[i]}, "
            f"Probabilities: {np.round(test_probabilities[i], 3)}"
        )

    # Basic checks.
    assert model.w.shape == (X_train.shape[1], 3)
    assert model.b.shape == (3,)
    assert test_probabilities.shape == (X_test.shape[0], 3)

    # Every softmax row should sum to approximately 1.
    assert np.allclose(test_probabilities.sum(axis=1), 1.0)

    # Training should generally reduce the loss.
    assert model.loss_history[-1] < model.loss_history[0]

    print("\nAll tests passed.")


if __name__ == "__main__":
    main()