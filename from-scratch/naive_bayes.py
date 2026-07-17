import numpy as np

class NaiveBayes:
    def __init__(self):
        self.class_labels = None
        self.class_priors = None
        self.means = None
        self.vars = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NaiveBayes":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).reshape(-1)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")

        if y.ndim != 1:
            raise ValueError("y must be a 1D array.")

        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must contain the same number of samples.")

        self.class_labels = np.unique(y)
        self.class_priors = np.zeros(len(self.class_labels), dtype=float)
        self.means = np.zeros((len(self.class_labels), X.shape[1]), dtype=float)
        self.vars = np.ones((len(self.class_labels), X.shape[1]), dtype=float)

        for i, label in enumerate(self.class_labels):
            X_class = X[y == label]
            self.class_priors[i] = X_class.shape[0] / X.shape[0]
            self.means[i] = np.mean(X_class, axis=0)
            self.vars[i] = np.var(X_class, axis=0) + 1e-6

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")

        log_probs = []
        for i in range(len(self.class_labels)):
            diff = X - self.means[i]
            variance = self.vars[i]
            log_prob = -0.5 * np.sum(np.log(2 * np.pi * variance), axis=1)
            log_prob -= 0.5 * np.sum((diff ** 2) / variance, axis=1)
            log_probs.append(log_prob)

        log_prob_matrix = np.vstack(log_probs).T
        log_prior = np.log(self.class_priors)
        log_post = log_prob_matrix + log_prior

        log_post -= np.max(log_post, axis=1, keepdims=True)
        probabilities = np.exp(log_post)
        probabilities /= np.sum(probabilities, axis=1, keepdims=True)

        return probabilities

    def predict(self, X: np.ndarray) -> np.ndarray:
        probabilities = self.predict_proba(X)
        return self.class_labels[np.argmax(probabilities, axis=1)]
        