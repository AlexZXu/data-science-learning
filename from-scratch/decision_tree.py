import numpy as np
class DecisionTree():
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X
        self.y = y

    def process_split(data: np.ndarray, split_index: int, split_value: float) -> tuple[list[int], list[int]]:
        left_indices = []
        right_indices = []
        for i in range(data.shape[0]):
            if (data[i][split_index] < split_value):
                left_indices.append(i)
            else:
                right_indices.append(i)

        return (left_indices, right_indices)
    

    def iterate_possible_splits(self, data: np.ndarray, y: np.ndarray):
        current_impurity = self.leaf_gini_impurity(y)
        lowest_impurity = np.inf
        
        left_data, right_data, split_index, split_value = None

        for i in range(data.shape[1]):
            sorted_indices = data[:, i].argsort()
            sorted_data = data[sorted_indices]
            sorted_y = y[sorted_indices]
            
            for j in range(1, data.shape[0]):          
                if (data[j][i] == data[j-1][i]):
                    continue
                
                left_y = sorted_y[:j] 
                right_y = sorted_y[j:]

                total_impurity = self.leaf_gini_impurity(left_y) + self.leaf_gini_impurity(right_y)

                print(total_impurity)
                if (total_impurity < lowest_impurity):
                    left_data = sorted_data[:j]
                    right_data = sorted_data[j:]
                    split_index = i
                    split_value = (data[j][i] + data[j-1][i]) / 2

        

                

    #impurity method for classification trees
    @staticmethod
    def leaf_gini_impurity(outputs: np.ndarray) -> float:
        (values, counts) = np.unique(outputs, return_counts = True)
        counts = counts / np.sum(counts, keepdims=True)
        return 1 - np.sum(counts**2)

    #impurity method for regression trees
    @staticmethod
    def leaf_ssr(outputs: np.ndarray) -> float:
        outputs -= np.average(outputs, keepdims=True)
        return np.sum(outputs**2)

class Node:
        def __init__(self, data=None, children=None, split_on=None, pred_class=None, is_leaf=False):
            self.data = data
            self.children = children
            self.split_on = split_on
            self.pred_class = pred_class