import numpy as np
import pyro
import pyro.infer
import pyro.optim
from bundesliga import settings
from bundesliga.model.base_footballmodel import FootballModel


class PyroModel(FootballModel):
    """
    A base class for Pyro-based probabilistic models to predict football match outcomes.

    This class provides the foundational structure for training and predicting using Pyro,
    a probabilistic programming framework built on PyTorch. It includes methods for training
    the model, calculating win probabilities, and validating predictions.

    Attributes:
        team_lexicon (dict): A dictionary mapping team names to unique IDs.
        model_config (dict): Configuration parameters for the model.
        sampler_config (dict): Configuration parameters for the sampler.
    """

    def __init__(self, team_lexicon, model_options):
        """
        Initializes the PyroModel with team lexicon and configuration parameters.

        Args:
            team_lexicon (dict): A dictionary mapping team names to unique IDs.
            parameters (dict): A dictionary containing model and sampler configurations.
        """
        self.team_lexicon = team_lexicon
        self.model_config = model_options["model_config"]
        self.sampler_config = model_options["sampler_config"]

    def train(self, X, y, parameters):
        """
        Trains the model using the provided data.

        Args:
            X (pd.DataFrame): Input data containing home and away team IDs.
            y (pd.DataFrame): Target data containing home goals, away goals, and match results (toto).
            parameters (dict): Additional parameters for training.

        Returns:
            list: A list of losses during training.
        """
        # pyro.clear_param_store()
        pyro.set_rng_seed(settings.SEED)
        np.random.seed(settings.SEED)

        home_id, away_id, home_goals, away_goals, toto = (
            X["home_id"].values,
            X["away_id"].values,
            y["home_goals"].values,
            y["away_goals"].values,
            y["toto"].values,
        )
        adam_params = {
            "lr": self.model_config["learning_rate"],
            "betas": self.model_config["betas"],
        }
        optimizer = pyro.optim.Adam(adam_params)
        self.model = self.get_model()
        self.guide = self.get_guide()
        svi = pyro.infer.SVI(
            model=self.model,
            guide=self.guide,
            optim=optimizer,
            loss=pyro.infer.Trace_ELBO(),
        )

        losses = []

        for t in range(len(self.team_lexicon)):
            loss = svi.step(home_id, away_id, home_goals, away_goals, toto)
            losses.append(loss)
            if t % 100 == 0:
                print(t, "\t", loss)

        return losses

    def get_probs_winner_from_goal_results(self, goals_of_team_1, goals_of_team_2):
        """
        Calculates the probabilities of team 1 winning, team 2 winning, or a draw based on goal results.

        Args:
            goals_of_team_1 (np.array): Goals scored by team 1.
            goals_of_team_2 (np.array): Goals scored by team 2.

        Returns:
            np.array: An array containing probabilities for team 1 win, team 2 win, and draw.
        """
        team1_wins = goals_of_team_1 > goals_of_team_2
        team2_wins = goals_of_team_1 < goals_of_team_2
        tie = goals_of_team_1 == goals_of_team_2

        p1 = team1_wins.mean()
        p2 = team2_wins.mean()
        tie = tie.mean()
        np.testing.assert_almost_equal(1.0, p1 + tie + p2)
        return np.array([p1, p2, tie])

    def predict_toto_probabilities(self, predictions, **kwargs):
        """
        Predicts the probabilities for match outcomes (toto) based on predicted goals.

        Args:
            predictions (dict): A dictionary containing predicted home and away goals.
            **kwargs: Additional keyword arguments.

        Returns:
            np.array: An array containing the predicted probabilities for match outcomes.
        """
        predicted_probabilities = self.get_probs_winner_from_goal_results(
            goals_of_team_1=predictions["home_goals"],
            goals_of_team_2=predictions["away_goals"],
        )
        result = np.array([predicted_probabilities])

        self._validate_output(result)
        return result
