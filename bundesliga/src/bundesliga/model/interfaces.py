from abc import abstractmethod
from typing import Union

import numpy as np
import pandas as pd
import pymc as pm
from pymc_experimental.model_builder import ModelBuilder


class pymc_FootballModel(ModelBuilder):
    # Give the model a name
    _model_type = "FootballModel"

    # And a version
    version = "0.1"

    @abstractmethod
    def build_model(self, X: pd.DataFrame, goals: pd.Series, **kwargs):
        """
        build_model creates the PyMC model

        Parameters:
        model_config: dictionary
            it is a dictionary with all the parameters that we need in our model example:  a_loc, a_scale, b_loc
        X : pd.DataFrame
            The input data that is going to be used in the model. This should be a DataFrame
            containing the features (predictors) for the model. For efficiency reasons, it should
            only contain the necessary data columns, not the entire available dataset, as this
            will be encoded into the data used to recreate the model.

        y : pd.Series
            The target data for the model. This should be a Series representing the output
            or dependent variable for the model.

        kwargs : dict
            Additional keyword arguments that may be used for model configuration.
        """
        raise NotImplementedError

    def _data_setter(
        self,
        X: Union[pd.DataFrame, pd.Series],
        y: Union[pd.Series, np.ndarray] = None,
    ) -> None:
        if isinstance(X, pd.DataFrame):
            x_data_home_values = X["home_id"].values
            x_data_away_values = X["away_id"].values
        else:
            # # Assuming "input" is the first column
            # x_values = X[:, 0]
            x_data_home_values = X[:, 0]
            x_data_away_values = X[:, 1]
        input_length = len(x_data_home_values)
        match_length = len(self.model_coords["match"])
        new_match_range = np.arange(input_length) + match_length
        dummy = np.ones(input_length, dtype=int)

        with self.model:
            self.model_coords["match"] = new_match_range
            pm.set_data(
                {"x_data_home": x_data_home_values},
                coords={"match": new_match_range},
            )
            pm.set_data(
                {"x_data_away": x_data_away_values},
                coords={"match": new_match_range},
            )

            if y is not None:
                pm.set_data(
                    {"goals": y.values if isinstance(y, pd.Series) else y},
                    coords={"match": new_match_range},
                )
            else:
                pm.set_data(
                    {"y_data_home": dummy},
                    coords={"match": new_match_range},
                )
                pm.set_data(
                    {"y_data_away": dummy},
                    coords={"match": new_match_range},
                )

    @staticmethod
    def get_default_model_config() -> dict:
        """
        Returns a class default config dict for model builder if no model_config is provided on class initialization.
        The model config dict is generally used to specify the prior values we want to build the model with.
        It supports more complex data structures like lists, dictionaries, etc.
        It will be passed to the class instance on initialization, in case the user doesn't provide any model_config of their own.
        """
        model_config = {
            "prior_params": {
                "off_mu": 1.75,
                "off_tau": 1.0,
                "def_mu": 0.0,
                "def_tau": 1.0,
            },
            "features": ["home_id", "away_id"],
            "targets": ["home_goals", "away_goals"],
        }

        return model_config
        # raise ValueError(
        #     "Model config not present. Add a model config to the parameters_data_science.yml."
        # )

    @staticmethod
    def get_default_sampler_config() -> dict:
        """
        Returns a class default sampler dict for model builder if no sampler_config is provided on class initialization.
        The sampler config dict is used to send parameters to the sampler .
        It will be used during fitting in case the user doesn't provide any sampler_config of their own.
        """
        sampler_config = {
            "chains": 2,
            "draws": 2000,
            "tune": 200,  # toDo: 10% of draws,
            "target_accept": 0.85,
            "random_seed": 42,
        }
        return sampler_config
        # raise ValueError(
        #     "Sampler config not present. Add a sampler config to the parameters_data_science.yml."
        # )

    @property
    def output_var(self):
        return "home_away_goals"

    @property
    def _serializable_model_config(self) -> dict[str, Union[int, float, dict]]:
        """
        _serializable_model_config is a property that returns a dictionary with all the model parameters that we want to save.
        as some of the data structures are not json serializable, we need to convert them to json serializable objects.
        Some models will need them, others can just define them to return the model_config.
        """
        return self.model_config

    def _save_input_params(self, idata) -> None:
        """
        Saves any additional model parameters (other than the dataset) to the idata object.

        These parameters are stored within `idata.attrs` using keys that correspond to the parameter names.
        If you don't need to store any extra parameters, you can leave this method unimplemented.

        Example:
            For saving customer IDs provided as an 'customer_ids' input to the model:
            self.customer_ids = customer_ids.values #this line is done outside of the function, preferably at the initialization of the model object.
            idata.attrs["customer_ids"] = json.dumps(self.customer_ids.tolist())  # Convert numpy array to a JSON-serializable list.
        """
        pass

        pass

    def _generate_and_preprocess_model_data(
        self,
        X: Union[pd.DataFrame, pd.Series],
        goals: Union[pd.Series, np.ndarray],
    ) -> None:
        """
        Depending on the model, we might need to preprocess the data before fitting the model.
        all required preprocessing and conditional assignments should be defined here.
        """

        def _get_teams(df: pd.DataFrame) -> np.array:
            teams, uniques = pd.factorize(
                X[["home_id", "away_id"]].values.flatten(), sort=True
            )

            return teams, uniques

        if isinstance(goals, pd.Series | np.ndarray):
            goals = goals.tolist()

        self.model_coords = {
            "team": _get_teams(X)[1],
            "match": np.arange(len(X)),
            "id": ["home_id", "away_id"],
            "goals": ["home_goals", "away_goals"],
        }
        self.X = X
        self.y = pd.DataFrame(goals, columns=["home_goals", "away_goals"])

    def predict_both_goals(
        self,
        X_pred: np.ndarray | pd.DataFrame | pd.Series,
        idata,
        extend_idata: bool = False,
        **kwargs,
    ) -> pd.DataFrame:
        with self.model:
            predictions = self.sample_posterior_predictive(
                X_pred, extend_idata, combined=False, **kwargs
            )
        # with self.model:
        #     predictions = pm.sample_posterior_predictive(
        #         idata, predictions=True, extend_inferencedata=False
        #     ).predictions
        home_mean = predictions["home_goals"].mean(dim=["chain", "draw"]).data
        away_mean = predictions["away_goals"].mean(dim=["chain", "draw"]).data
        return pd.DataFrame(
            {
                "home_goals": home_mean,
                "away_goals": away_mean,
            }
        )


class pyro_FootballModel:
    # Give the model a name
    _model_type = "FootballModel"

    # And a version
    version = "0.1"

    pass
