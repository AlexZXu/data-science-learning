import numpy as np

class LogisticRegression():
    def __init__(self,
                 lr: float=0.01, 
                 max_iter: int = 10_000,
                 tol: float=1e-8,
                 reg=None,
                 delta=0.1):
        self.lr = lr
        self.max_iter = max_iter
        self.tol = tol
        self.reg = reg
        self.delta = delta
        
        self.w = None
        self.b = None
        self.loss_history = []

    @staticmethod
    def one_hot_Y(y: np.ndarray, C: int) -> np.ndarray:
        one_hot_Y = np.zeros(shape=(y.shape[0], C))
        one_hot_Y[np.arange(y.shape[0]), y] = 1
        return one_hot_Y

    @staticmethod
    def validate_inputs(X: np.ndarray, y: np.ndarray):
        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")

        if y.ndim != 1:
            raise ValueError("y must be a 1D array.")

        if X.shape[0] != y.shape[0]:
            raise ValueError(
                "X and y must contain the same number of samples."
            )

    def fit(self, X: np.ndarray, y: np.ndarray, C: int) -> "LogisticRegression":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int).reshape(-1)
        
        Y_hot = self.one_hot_Y(y, C)

        self.validate_inputs(X, y)
        
        n_features = X.shape[1]
    
        self.w = np.zeros(shape=(n_features, C), dtype=float)
        self.b = np.zeros(shape=(C), dtype=float)
        self.loss_history = []

        previous_loss = np.inf

        for _ in range(self.max_iter):
            y_pred = self.predict(X)
            loss = self.loss(y_pred, y)

            (grad_w, grad_b) = self.grad_loss(X, y_pred, Y_hot)

            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b
            
            self.loss_history.append(loss)

            improvement = previous_loss - loss
            if abs(improvement) < self.tol:
                break
            
            previous_loss = loss

        return self


    def predict(self, X: np.ndarray) -> np.ndarray:
        logits = X @ self.w + self.b

        logits -= np.max(logits, axis=1, keepdims=True)

        exp_logits = np.exp(logits)
        sum_rows = np.sum(exp_logits, axis=1, keepdims=True)

        return exp_logits / sum_rows

    
    @staticmethod
    def loss(y_pred: np.ndarray, y: np.ndarray) -> float:
        correct_class_probs = y_pred[np.arange(y.shape[0]), y]
        correct_prob_logged = np.log(correct_class_probs + 1e-15)
        return -np.mean(correct_prob_logged)

    def grad_loss(self, X, y_pred, y_hot) -> tuple[np.ndarray, float]:    
        error = y_pred - y_hot

        grad_w = X.T @ error / X.shape[0] 
        grad_b = np.mean(error, axis=0)

        if self.reg == "l1":
            grad_w += self.delta * np.sign(self.w)
        elif self.reg == "l2":
            grad_w += 2 * self.delta * self.w    

        return grad_w, grad_b
