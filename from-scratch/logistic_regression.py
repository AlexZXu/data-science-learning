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
        
        self.w = None
        self.b = None
        self.loss_history = []

    def fit(self, X: np.ndarray, y: np.ndarray, C: int) -> "LogisticRegression":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")

        if y.ndim != 1:
            raise ValueError("y must be a 1D array.")

        if X.shape[0] != y.shape[0]:
            raise ValueError(
                "X and y must contain the same number of samples."
            )
        
        n_features = X.shape[1]
    
        self.w = np.zeros(shape=(n_features, C), dtype=float)
        self.b = np.zeros(C, dtype=float)
        self.loss_history = []

        previous_loss = np.inf

        for _ in range(self.max_iter):
            y_hat = self.predict(X)
            loss = self.loss(y_hat, y)

            (grad_w, grad_b) = self.grad_loss(X, y_hat, y)

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

        exp_logits = np.exp(logits)
        sum_rows = np.sum(exp_logits, axis=1, keepdims=True)

        return exp_logits / sum_rows

    
    @staticmethod
    def loss(y_hat: np.ndarray, y: np.ndarray) -> float:
        correct_class_probs = y_hat[np.arange(y.shape[0]), y]

        return -np.mean(correct_class_probs + 1e-15)

    def grad_loss(self, X, y_hat, y) -> tuple[np.ndarray, float]:    
        error = y_hat - y

        grad_w = X.T @ error / X.shape[0]
        grad_b = np.mean(error)

        return grad_w, grad_b
