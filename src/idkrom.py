import numpy as np
import pandas as pd
from abc import ABC, abstractmethod


class idkROM(ABC):

    def __init__(self):
        pass

    """Clase base abstracta
    Sirve de plantilla para las subclases de cada método ROM, donde se implementan las funciones train y evaluate dedicadas."""
    class Modelo(ABC):
        def __init__(self):
            self.model = None

        @abstractmethod
        def train(self, X_train, y_train, X_val, y_val):
            """Método a sobrescribir en cada subclase"""
            raise NotImplementedError
        
        @abstractmethod
        def predict(self, X_test):
            """Método a sobrescribir en cada subclase"""
            raise NotImplementedError
        
        @abstractmethod
        def evaluate(self, X_test, y_test, y_pred):
            """Método a sobrescribir en cada subclase"""
            raise NotImplementedError

        # search best hyperparams
        

