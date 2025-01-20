import numpy as np
import pyro
import pyro.distributions as dist
import pyro.infer
import pyro.optim
from bundesliga.model.model_interface import FootballModel


class PyroModel(FootballModel):
    def __init__(self, team_lexicon, parameters):
        self.team_lexicon = team_lexicon
        self.model_config = parameters["model_config"]
        self.sampler_config = parameters["sampler_config"]

    def train(self, X, y, parameters):
        pyro.clear_param_store()
        pyro.set_rng_seed(self.sampler_config["random_seed"])
        np.random.seed(self.sampler_config["random_seed"])

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
