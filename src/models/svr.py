from sklearn.svm import SVR
from src.idkROM_v0 import idkROM


"""SVR"""
class SVRROM(idkROM.Modelo):
    """Modelo basado en Support Vector Regression (SVR)."""
    def __init__(self, kernel='rbf', C=1.0, epsilon=0.1):
        super().__init__()
        self.model = SVR(kernel=kernel, C=C, epsilon=epsilon)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        return self.model.predict(X_test)