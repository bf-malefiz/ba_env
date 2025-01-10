from abc import ABC, abstractmethod


class FootballModelBase(ABC):
    @abstractmethod
    def build_model(self, X, goals, **kwargs):
        pass

    @abstractmethod
    def fit(self, **kwargs):
        pass

    @abstractmethod
    def predict(self, X, **kwargs):
        pass
