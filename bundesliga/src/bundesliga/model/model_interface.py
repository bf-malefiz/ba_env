from abc import ABC, abstractmethod


class FootballModel(ABC):
    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def predict_toto_probabilities(self, home_teams, away_teams):
        pass
