import numpy as np
from decision_tree import DecisionTree

def main():
    X = np.array([
        [1.0, 1.0],
        [1.5, 2.0],
        [2.0, 1.5],
        [2.5, 2.5],
        [6.0, 5.5],
        [6.5, 6.0],
        [7.0, 5.0],
        [7.5, 6.5],
    ])

    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])

    tree = DecisionTree(X, y, max_depth=3)
    root = tree.fit()





if (__name__ == "__main__"):
    main()