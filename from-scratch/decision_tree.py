from __future__ import annotations

import numpy as np


class Node:
    def __init__(self, X: np.ndarray, y: np.ndarray, depth: int = 0):
        self.X = X
        self.y = y
        self.depth = depth

        self.left: Node | None = None
        self.right: Node | None = None

        self.split_index: int | None = None
        self.split_value: float | None = None

        self.pred_class = None
        self.is_leaf = True

class DecisionTree:
    def __init__(self, max_depth: int = 8, min_impurity_decrease: float = 0.05):
        self.max_depth = max_depth
        self.min_impurity_decrease = min_impurity_decrease
        self.root: Node | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> DecisionTree:
        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")

        if y.ndim != 1:
            raise ValueError("y must be a 1D array.")

        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must contain the same number of samples.")

        self.root = Node(X, y)
        self._split_node(self.root)

        return self

    def _split_node(self, node: Node) -> None:
        # Stop if the node is pure, too small, or at maximum depth.
        if (node.X.shape[0] <= 1 or node.depth >= self.max_depth or self.gini_impurity(node.y) == 0):
            self._make_leaf(node)
            return

        split = self._find_best_split(node.X, node.y)

        if split is None:
            self._make_leaf(node)
            return

        (
            split_index,
            split_value,
            left_X,
            left_y,
            right_X,
            right_y,
            best_impurity,
        ) = split

        current_impurity = self.gini_impurity(node.y)
        impurity_decrease = current_impurity - best_impurity

        if (impurity_decrease < self.min_impurity_decrease):
            self._make_leaf(node)
            return

        node.is_leaf = False
        node.split_index = split_index
        node.split_value = split_value

        node.left = Node(left_X, left_y, depth=node.depth + 1)
        node.right = Node(right_X, right_y, depth=node.depth + 1)

        self._split_node(node.left)
        self._split_node(node.right)

    def _find_best_split(self, X: np.ndarray, y: np.ndarray) -> tuple | None:
        best_impurity = np.inf
        best_split = None

        n_samples, n_features = X.shape

        for feature_index in range(n_features):
            sorted_indices = np.argsort(X[:, feature_index])

            sorted_X = X[sorted_indices]
            sorted_y = y[sorted_indices]

            for i in range(1, n_samples):
                prev_value = sorted_X[i - 1, feature_index]
                curr_value = sorted_X[i, feature_index]

                if (prev_value == curr_value):
                    continue

                left_X = sorted_X[:i]
                left_y = sorted_y[:i]
                right_X = sorted_X[i:]
                right_y = sorted_y[i:]

                left_weight = i / n_samples
                right_weight = 1 - left_weight

                total_impurity = (
                    left_weight * self.gini_impurity(left_y)
                    + right_weight * self.gini_impurity(right_y)
                )

                if (total_impurity < best_impurity):
                    split_value = (prev_value + curr_value) / 2

                    best_impurity = total_impurity
                    best_split = (
                        feature_index,
                        split_value,
                        left_X,
                        left_y,
                        right_X,
                        right_y,
                        total_impurity,
                    )

        return best_split

    def _make_leaf(self, node: Node) -> None:
        node.is_leaf = True

        classes, counts = np.unique(node.y, return_counts=True)
        node.pred_class = classes[np.argmax(counts)]

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.root is None:
            raise RuntimeError("The tree must be fitted before prediction.")

        return np.array([self._predict_one(row) for row in X])

    def _predict_one(self, row: np.ndarray):
        node = self.root

        while not node.is_leaf:
            if node.split_index is None or node.split_value is None:
                raise RuntimeError("A non-leaf node is missing split information.")

            if row[node.split_index] < node.split_value:
                node = node.left
            else:
                node = node.right

            if node is None:
                raise RuntimeError("Tree structure is incomplete.")

        return node.pred_class

    @staticmethod
    def gini_impurity(y: np.ndarray) -> float:
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / counts.sum()

        return 1 - np.sum(probabilities**2)

    @staticmethod
    def ssr(y: np.ndarray) -> float:
        return np.sum((y - np.mean(y)) ** 2)