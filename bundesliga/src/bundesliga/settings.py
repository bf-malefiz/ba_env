"""Project settings. There is no need to edit this file unless you want to change values
from the Kedro defaults. For further information, including these default values, see
https://docs.kedro.org/en/stable/kedro_project_setup/settings.html."""

from utils import merge_dicts

# from hooks import NodeInputReplacementHook, PipelineParamHook

# Instantiated project hooks.
# For example, after creating a hooks.py and defining a ProjectHooks class there, do
# from pandas_viz.hooks import ProjectHooks

# Hooks are executed in a Last-In-First-Out (LIFO) order.
HOOKS = (
    # NodeInputReplacementHook(),
    # PipelineParamHook(),
)


# Installed plugins for which to disable hook auto-registration.
# DISABLE_HOOKS_FOR_PLUGINS = ("kedro-viz",)

from pathlib import Path  # noqa: E402

from kedro_viz.integrations.kedro.sqlite_store import SQLiteStore  # noqa: E402

# Class that manages storing KedroSession data.
SESSION_STORE_CLASS = SQLiteStore

# Setup for Experiment Tracking
# The SQLite DB required for experiment tracking is stored by default
# (supported from python >= 3.9 and Kedro-Viz 9.2.0) in the .viz folder
# of your project. To store it in another directory, provide the keyword argument
# `SESSION_STORE_ARGS` to pass to the `SESSION_STORE_CLASS` constructor.
SESSION_STORE_ARGS = {"path": str(Path(__file__).parents[2])}

# Directory that holds configuration.
CONF_SOURCE = "conf"

# Class that manages how configuration is loaded.
from kedro.config import OmegaConfigLoader  # noqa: E402

CONFIG_LOADER_CLASS = OmegaConfigLoader
# Keyword arguments to pass to the `CONFIG_LOADER_CLASS` constructor.
CONFIG_LOADER_ARGS = {
    "base_env": "base",
    "default_run_env": "local",
    "config_patterns": {
        "parameters": ["parameters*", "parameters*/**", "**/parameters*"],
        "datasets": ["datasets*", "datasets*/**", "**/datasets*"],
    },
    "custom_resolvers": {"merge": lambda x1, x2: merge_dicts(x1, x2)},
}

DYNAMIC_PIPELINES_MAPPING = {
    # "etl": ["base"],
    # "goal_predictor": ["base"],
    "pymc": ["f1"],
    # "pymc2": ["f1"],
}

# Class that manages Kedro's library components.
# from kedro.framework.context import KedroContext
# CONTEXT_CLASS = KedroContext


# Class that manages the Data Catalog.
# from kedro.io import DataCatalog
# DATA_CATALOG_CLASS = DataCatalog
