from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from src.idkROM_v0 import idkROM


"""Response Surface"""
class ResponseSurfaceROM(idkROM.Modelo):
    """Modelo basado en ajuste polinómico (Response Surface)."""
    def __init__(self, degree=2):
        super().__init__()
        self.degree = degree
        self.model = None
        self.poly = PolynomialFeatures(degree=self.degree)

    def train(self, X_train, y_train):
        X_poly = self.poly.fit_transform(X_train)
        self.model = LinearRegression()
        self.model.fit(X_poly, y_train)

    def evaluate(self, X_test, y_test):
        X_poly = self.poly.transform(X_test)
        return self.model.predict(X_poly)