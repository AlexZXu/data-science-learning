import numpy as np
from decision_tree import DecisionTree

def main():
    X = np.array([
        # Class 0: mostly lower-left
        [1.0, 1.2],
        [1.3, 2.1],
        [1.8, 1.5],
        [2.2, 2.4],
        [2.6, 1.1],
        [3.0, 2.0],

        # Class 1: mostly upper-left
        [1.1, 6.5],
        [1.5, 7.2],
        [2.0, 5.8],
        [2.4, 6.8],
        [2.8, 7.5],
        [3.2, 5.9],

        # Class 2: mostly right side
        [6.0, 1.0],
        [6.5, 2.5],
        [7.0, 4.0],
        [7.4, 5.5],
        [6.2, 7.0],
        [8.0, 6.5],

        # A few harder points
        [4.0, 2.0],
        [4.2, 6.5],
        [5.0, 3.5],
        [5.2, 5.5],
    ])

    y = np.array([
        0, 0, 0, 0, 0, 0,
        1, 1, 1, 1, 1, 1,
        2, 2, 2, 2, 2, 2,
        0, 1, 2, 2,
    ])

    tree = DecisionTree(max_depth=3)
    tree.fit(X, y)

    X_test = np.array([
        [1.5, 1.7],  # likely class 0
        [2.0, 6.7],  # likely class 1
        [7.0, 3.0],  # likely class 2
        [4.1, 1.8],  # probably class 0
        [4.3, 6.8],  # probably class 1
        [5.5, 4.5],  # probably class 2
    ])
    
    test = tree.predict(X_test)
    print(test)

if (__name__ == "__main__"):
    main()