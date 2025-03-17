from scipy.interpolate import RBFInterpolator
from src.idkrom import idkROM


"""RBF"""
class RBFROM(idkROM.Modelo):
    def __init__(self):
        super().__init__()
        self.model = None

    def train(self, X_train, y_train):
        self.model = RBFInterpolator(X_train, y_train)

    def evaluate(self, X_test, y_test):
        return self.model(X_test)