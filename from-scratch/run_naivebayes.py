import numpy as np
from naive_bayes import NaiveBayes


def main():
    X = np.array([
        [1.0, 1.0],
        [1.2, 0.9],
        [1.1, 1.1],
        [4.0, 4.0],
        [4.2, 3.8],
        [3.9, 4.1],
        [8.0, 8.0],
        [8.1, 7.9],
        [7.9, 8.2],
    ])

    y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])

    model = NaiveBayes()
    model.fit(X, y)

    test_points = np.array([
        [1.1, 1.0],
        [4.1, 4.0],
        [8.0, 8.1],
    ])

    predictions = model.predict(test_points)
    probabilities = model.predict_proba(test_points)

    print("Predictions:", predictions)
    print("Probabilities:")
    print(probabilities)


if __name__ == "__main__":
    main()
