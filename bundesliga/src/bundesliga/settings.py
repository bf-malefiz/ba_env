"""Project settings.

There is no need to edit this file unless you want to change values
from the Kedro defaults or add new Models. For further information, including these default values, see
https://docs.kedro.org/en/stable/kedro_project_setup/settings.html.

**`DYNAMIC_PIPELINES_MAPPING`**: Specifies the type of engine and model used. \ni.e.: {"pyro": ["simple", "toto"]}
"""

from pathlib import Path

import yaml
from kedro.config import OmegaConfigLoader
from kedro_viz.integrations.kedro.sqlite_store import SQLiteStore  # noqa: E402

from bundesliga.hooks.model_tracking_hooks import ModelTrackingHooks
from bundesliga.utils.utils import merge_dicts

# -----------------------------------------------------------------------------
# Random Seed
# -----------------------------------------------------------------------------

# Set the random seed for reproducibility
SEED = 42

# -----------------------------------------------------------------------------
# Hooks
# -----------------------------------------------------------------------------

# Instantiate project hooks
# Hooks are executed in a Last-In-First-Out (LIFO) order.
HOOKS = (ModelTrackingHooks(),)

# -----------------------------------------------------------------------------
# Session Store
# -----------------------------------------------------------------------------

# Class that manages storing KedroSession data
SESSION_STORE_CLASS = SQLiteStore

# Keyword arguments to pass to the `SESSION_STORE_CLASS` constructor
# The SQLite DB required for experiment tracking is stored by default
# (supported from python >= 3.9 and Kedro-Viz 9.2.0) in the.viz folder
# of your project.
# To store it in another directory, provide the keyword argument
# `SESSION_STORE_ARGS` to pass to the `SESSION_STORE_CLASS` constructor.
SESSION_STORE_ARGS = {"path": str(Path(__file__).parents[2])}

# -----------------------------------------------------------------------------
# Configuration Loader
# -----------------------------------------------------------------------------

# Directory that holds configuration.
CONF_SOURCE = "conf"

# Class that manages how configuration is loaded.
CONFIG_LOADER_CLASS = OmegaConfigLoader

# Keyword arguments to pass to the `CONFIG_LOADER_CLASS` constructor.
CONFIG_LOADER_ARGS = {
    "base_env": "base",
    "default_run_env": "local",
    # Define the configuration patterns to use
    "config_patterns": {
        "parameters": ["parameters*", "parameters*/**", "**/parameters*"],
        "datasets": ["datasets*", "datasets*/**", "**/datasets*"],
    },
    # Add a custom resolver for merging dictionaries
    "custom_resolvers": {"merge": lambda x1, x2: merge_dicts(x1, x2)},
}

# -----------------------------------------------------------------------------
# Dynamic Pipelines
# -----------------------------------------------------------------------------

# Define the mapping for dynamic pipelines
DYNAMIC_PIPELINES_MAPPING = {
    "pymc": ["simple"],
    "pyro": ["simple", "simple2"],
}

# -----------------------------------------------------------------------------
# Datasets
# -----------------------------------------------------------------------------

# Exctract the datasets from the YAML file
base_path = Path(__file__).resolve().parent.parent.parent
yaml_file = base_path / "conf" / "base" / "datasets_list.yml"
with open(yaml_file, encoding="utf-8") as file:
    data = yaml.safe_load(file)

# Store the list of datasets
DATASETS = data.get(
    "datasets",
)

# -----------------------------------------------------------------------------
# Context and Catalog
# -----------------------------------------------------------------------------

# Class that manages Kedro's library components.
# from kedro.framework.context import KedroContext
# CONTEXT_CLASS = KedroContext

# Class that manages the Data Catalog.
# from kedro.io import DataCatalog
# DATA_CATALOG_CLASS = DataCatalog
