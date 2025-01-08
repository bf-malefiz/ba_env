import arviz as az
import pandas as pd
from modelbuilder import FootballModel


def predict(idata, x_data, parameters):
    x_data = pd.DataFrame(data={"home_team": [1, 4], "away_team": [2, 6]})

    fname = "S:/___Studium/Bachelor_Arbeit/ba_env/bundesliga/data/06_models/model_1_idata_active.nc"
    model = FootballModel().load(fname)
    with model:
        idata = model.sample_posterior_predictive(
            x_data, var_names=["home_goals", "away_goals"]
        )

        return az.convert_to_dataset(idata)
