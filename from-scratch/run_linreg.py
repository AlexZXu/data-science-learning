import numpy as np

from linear_regression import LinearRegression


def main():
    rng = np.random.default_rng(seed=42)

    # True relationship:
    # y = 3x1 - 2x2 + 5
    true_w = np.array([6, -5.0])
    true_b = 5.0

    n_samples = 500
    X = rng.normal(size=(n_samples, 2))

    noise = rng.normal(loc=0.0, scale=0.1, size=n_samples)
    y = X @ true_w + true_b + noise

    model = LinearRegression(
        lr=0.05,
        max_iter=10_000,
        tol=1e-10,
    )

    model.fit(X, y)

    y_hat = model.predict(X)
    mse = np.mean((y_hat - y) ** 2)

    print("True weights:      ", true_w)
    print("Estimated weights: ", model.w)
    print()

    print("True intercept:      ", true_b)
    print("Estimated intercept: ", model.b)
    print()

    print("Training MSE:", mse)
    print("Iterations:", len(model.loss_history))
    print("Initial loss:", model.loss_history[0])
    print("Final loss:", model.loss_history[-1])

    # Basic correctness checks
    assert model.w.shape == true_w.shape
    assert np.allclose(model.w, true_w, atol=0.05)
    assert np.isclose(model.b, true_b, atol=0.05)
    assert mse < 0.02
    assert model.loss_history[-1] < model.loss_history[0]

    # Test predictions on new observations
    X_new = np.array([
        [1.0, 2.0],
        [-1.0, 3.0],
        [0.0, 0.0],
    ])

    predictions = model.predict(X_new)
    expected = X_new @ true_w + true_b

    print("\nNew predictions:")
    print(predictions)

    print("\nExpected values without noise:")
    print(expected)

    assert np.allclose(predictions, expected, atol=0.1)

    print("\nAll tests passed.")


if __name__ == "__main__":
    main()