"""
Module: pymc_model.py

Summary:
    Base class for PyMC-based football match prediction models.

Dependencies:
    - abc.abstractmethod
    - ast.Dict
    - typing.Optional, Union
    - numpy as np
    - pandas as pd
    - pymc as pm
    - pymc_experimental.model_builder.ModelBuilder
    - bundesliga.settings
    - bundesliga.model.base_footballmodel.FootballModel
"""

from abc import abstractmethod
from ast import Dict
from typing import Optional, Union

import numpy as np
import pandas as pd
import pymc as pm
from pymc_experimental.model_builder import ModelBuilder

from bundesliga import settings
from bundesliga.model.base_footballmodel import FootballModel


class PymcModel(ModelBuilder, FootballModel):
    """
    Base class for PyMC-based football match prediction models.

    Provides a simple framework to build, train, and predict football match outcomes using PyMC.
    Inherits from ModelBuilder and FootballModel, and includes methods for configuration, data handling,
    and model evaluation.
    """

    def __init__(
        self,
        model_options: Optional[Dict] = None,
        team_lexicon: Optional[Dict] = None,
    ):
        """
        Initializes the PyMC model with default and custom configurations.

        Sets up the model and sampler configurations and assigns the team lexicon used for mapping team identifiers.
        Custom configurations provided in model_options update the defaults.

        Args:
            model_options (Optional[Dict], optional): Custom configuration options for the model.
            team_lexicon (Optional[Dict], optional): Mapping of team identifiers to their corresponding values.
        """
        if "model_config" in model_options:
            self.model_config = model_options["model_config"]
        else:
            self.model_config = self.get_default_model_config()
            print("Initializing with default model config")
        if "sampler_config" in model_options:
            self.sampler_config = model_options["sampler_config"]
        else:
            self.sampler_config = self.get_default_sampler_config()
            print("Initializing with default sampler config")

        self.team_lexicon = team_lexicon
        super().__init__(self.model_config, self.sampler_config)

    _model_type = "FootballModel"
    version = "0.1"

    @abstractmethod
    def build_model(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        """
        Constructs the PyMC model using the provided data.

        Subclasses must implement this method to define the model's structure.

        Args:
            X (pd.DataFrame): Input features, expected to include 'home_id' and 'away_id' columns.
            y (pd.Series): Target data representing outcomes (e.g., goals).
            **kwargs: Additional configuration parameters.
        """
        raise NotImplementedError

    def _data_setter(
        self,
        X: Union[pd.DataFrame, pd.Series],
        y: Union[pd.Series, np.ndarray] = None,
    ) -> None:
        """
        Sets the data for the PyMC model.

        This method updates the model's data containers with new input and target data.

        Args:
            X (Union[pd.DataFrame, pd.Series]): Input features for the model.
            y (Union[pd.Series, np.ndarray], optional): Target data for the model. Defaults to None.

        Raises:
            ValueError: If the input data is not in the expected format.
        """
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
        dummy = np.ones(shape=(1, 2), dtype=int)
        dummy_toto = np.ones(len(new_match_range), dtype=int)
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
                    {"y": y.values if isinstance(y, pd.Series) else y},
                    coords={"match": new_match_range},
                )
            else:
                pm.set_data(
                    {"y_data": dummy},
                    coords={
                        "match": new_match_range,
                        "y": ["home_goals", "away_goals"],
                    },
                )
                if "toto_data" in self.model.named_vars:
                    pm.set_data(
                        {"toto_data": dummy_toto},
                        coords={
                            "match": new_match_range,
                        },
                    )

    @staticmethod
    def get_default_model_config() -> dict:
        """
        Returns a class default config dict for model builder if no model_config is provided on class initialization.
        The model config dict is generally used to specify the prior values we want to build the model with.
        It supports more complex data structures like lists, dictionaries, etc.
        It will be passed to the class instance on initialization, in case the user doesn't provide any model_config of their own.

        Returns:
            dict: Default model configuration.
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

    @staticmethod
    def get_default_sampler_config() -> dict:
        """
        Returns a class default sampler dict for model builder if no sampler_config is provided on class initialization.
        The sampler config dict is used to send parameters to the sampler .
        It will be used during fitting in case the user doesn't provide any sampler_config of their own.

        Returns:
        dict: Default sampler configuration.
        """
        sampler_config = {
            "chains": 2,
            "draws": 2000,
            "tune": 200,  # toDo: 10% of draws,
            "target_accept": 0.85,
            "random_seed": 42,
        }
        return sampler_config

    @property
    def output_var(self):
        """
        Returns the name of the output variable for the model.

        Returns:
            str: The name of the output variable (e.g., "goals").
        """
        return "goals"

    @property
    def _serializable_model_config(self) -> dict[str, Union[int, float, dict]]:
        """
        _serializable_model_config is a property that returns a dictionary with all the model parameters that we want to save.
        as some of the data structures are not json serializable, we need to convert them to json serializable objects.
        Some models will need them, others can just define them to return the model_config.
        Returns:
            dict: A dictionary containing the serializable model configuration.
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

    def _generate_and_preprocess_model_data(
        self,
        X: Union[pd.DataFrame, pd.Series],
        y: Union[pd.Series, np.ndarray],
    ) -> None:
        """
        Preprocesses the input data and sets up the model coordinates.

        Extracts team information, creates a range for matches, and stores the processed input features and targets
        for use in the PyMC model.

        Args:
            X (Union[pd.DataFrame, pd.Series]): Input features for the model.
            y (Union[pd.Series, np.ndarray]): Target data for the model.
        """

        if isinstance(y, pd.Series | np.ndarray):
            y = y.tolist()

        self.model_coords = {
            "team": self.team_lexicon.index,
            "match": np.arange(len(X)),
            "X": ["home_id", "away_id"],
            "y": ["home_goals", "away_goals"],
        }
        self.X = X
        self.y = y

    def train(self, X, y, **kwargs):
        """
        Trains the PymcModel using the provided data.

        Prepares the target data and fits the model using the configured sampler settings.

        Args:
            X: Input features for training.
            y: Target data for training.
            **kwargs: Additional keyword arguments for training.

        Returns:
            idata: The inference data object containing the results of the training.
        """
        goals = y.apply(lambda row: (row["home_goals"], row["away_goals"]), axis=1)
        self.toto = y["toto"].values
        idata = self.fit(
            X=X,
            y=goals,
            random_seed=settings.SEED,
            nchains=self.sampler_config["chains"],
            draws=self.sampler_config["draws"],
        )
        return idata

    def predict_goals(
        self,
        test_data: np.ndarray | pd.DataFrame | pd.Series,
        extend_idata: bool = False,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Predicts the goals for home and away teams using the trained model.

        Args:
            test_data (Union[np.ndarray, pd.DataFrame, pd.Series]): Input data for predictions.
            extend_idata (bool, optional): Whether to extend the inference data. Defaults to False.
            **kwargs: Additional keyword arguments for prediction.

        Returns:
            pd.DataFrame: A DataFrame containing the predicted goals for home and away teams.
        """
        pp = self.predict_posterior(
            X_pred=test_data,
            combined=True,
            random_seed=settings.SEED,
            **kwargs,
        )

        home_goals = pp.isel(y=0).values.flatten()
        away_goals = pp.isel(y=1).values.flatten()
        # tie = home_goals == away_goals
        df = pd.DataFrame({"home_goals": home_goals, "away_goals": away_goals})
        return df

    def predict_toto_probabilities(
        self,
        predictions,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Predicts the probabilities of match outcomes (toto) based on predicted goals.

        Args:
            predictions (pd.DataFrame): A DataFrame containing predicted goals for home and away teams.
            **kwargs: Additional keyword arguments for prediction.

        Returns:
            pd.DataFrame: A DataFrame containing the probabilities for:
                - Team 1 winning (index 0),
                - Team 2 winning (index 1),
                - A draw (index 2).
        """
        team1_wins = predictions["home_goals"] > predictions["away_goals"]
        team2_wins = predictions["home_goals"] < predictions["away_goals"]
        tie = predictions["home_goals"] == predictions["away_goals"]
        p1 = team1_wins.mean()
        p2 = team2_wins.mean()
        tie = tie.mean()
        np.testing.assert_almost_equal(1, p1 + tie + p2)
        result = np.array([[p1, p2, tie]])
        self._validate_output(result)
        return result

    def _validate_input(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Validates the input data before model training or prediction.

        Args:
            X (pd.DataFrame): Input features.
            y (pd.Series): Target data.

        Raises:
            ValueError: If required columns are missing or data is invalid.
        """

        required_columns = ["home_id", "away_id"]
        if not all(col in X.columns for col in required_columns):
            raise ValueError(f"Input data must contain columns: {required_columns}")

        if y is None:
            raise ValueError("Target data (y) must be provided.")
