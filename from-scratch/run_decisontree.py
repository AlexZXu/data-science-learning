import numpy as np
from decision_tree import DecisionTree

def main():
    dt = DecisionTree()

    print(DecisionTree.leaf_gini_impurity([2, 2, 1, 2, 3, 1, 1]))

    print(DecisionTree.leaf_ssr([12.4, 53.4, 24.23, 69.4]))

if (__name__ == "__main__"):
    main()