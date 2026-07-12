import numpy as np
class DecisionTree():
    def __init__(self, X: np.ndarray, y: np.ndarray, max_depth = 8):
        self.X = X
        self.y = y
        self.root = Node(X, y)
        self.max_depth = max_depth

    def fit(self):
        curr_leaves = [self.root]
        new_leaves = []

        for i in range(0, self.max_depth):
            for j in range(len(curr_leaves)):
                leaf = curr_leaves[j]

                if (leaf.X.shape[0] == 1):
                    continue
                
                current_impurity = self.leaf_gini_impurity(leaf.y)
                
                left_data, right_data, split_index, split_value, f_left_y, f_right_y, lowest_impurity = self.iterate_possible_splits(leaf.X, leaf.y)

                print(left_data, right_data, split_index, split_value, lowest_impurity)
                if (current_impurity - lowest_impurity < 0.05):
                    continue

                leaf.split_index = split_index
                leaf.split_value = split_value

                leaf.is_leaf = False

                new_left_leaf = Node(left_data, f_left_y)
                new_right_leaf = Node(right_data, f_right_y)
                
                leaf.children = [new_left_leaf, new_right_leaf]
                new_leaves.extend([new_left_leaf, new_right_leaf])

            curr_leaves = new_leaves

        return self.root

    """
    Method to test all possible splits for some input data and train outputs
    """
    def iterate_possible_splits(self, data: np.ndarray, y: np.ndarray):
        lowest_impurity = np.inf

        left_data = right_data = split_index =split_value = None
        f_left_y = f_right_y = None

        for i in range(data.shape[1]):
            sorted_indices = data[:, i].argsort()
            sorted_data = data[sorted_indices]
            sorted_y = y[sorted_indices]
            
            for j in range(1, data.shape[0]):          
                if (sorted_data[j][i] == sorted_data[j-1][i]):
                    continue
                
                left_y = sorted_y[:j] 
                right_y = sorted_y[j:]

                total_impurity = (j / sorted_y.shape[0]) * self.leaf_gini_impurity(left_y) + (1 - (j / sorted_y.shape[0])) * self.leaf_gini_impurity(right_y)

                if (total_impurity < lowest_impurity):
                    left_data = sorted_data[:j]
                    right_data = sorted_data[j:]
                    split_index = i
                    split_value = (data[j][i] + data[j-1][i]) / 2
                    f_left_y = left_y
                    f_right_y = right_y
                    lowest_impurity = total_impurity

        return left_data, right_data, split_index, split_value, f_left_y, f_right_y, lowest_impurity

    """
    Impurity method for classification trees
    """
    @staticmethod
    def leaf_gini_impurity(outputs: np.ndarray) -> float:
        (values, counts) = np.unique(outputs, return_counts = True)
        counts = counts / np.sum(counts, keepdims=True)
        return 1 - np.sum(counts**2)

    """
    Impurity method for regression trees
    """
    @staticmethod
    def leaf_ssr(outputs: np.ndarray) -> float:
        outputs -= np.average(outputs, keepdims=True)
        return np.sum(outputs**2)


class Node:
    def __init__(self, X:np.ndarray=None, y=None, children=None, split_index=None, split_value=None, pred_class=None, is_leaf=True):
        self.X = X
        self.y = y
        self.children = children
        self.split_index = split_index
        self.split_value = split_value
        self.pred_class = pred_class
        self.is_leaf = is_leaf

    