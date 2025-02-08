"""
===============================================================================
               TEST DOCUMENTATION: Kedro Data Processing Pipeline
===============================================================================
Title:
    Integration Tests for the Bundesliga Data Processing Pipeline

Purpose:
    This test suite verifies that each node (and combinations of nodes) within
    the data processing pipeline runs successfully in a Kedro environment,
    respecting namespaced sub-pipelines for multiple datasets. Each test
    focuses on one or more nodes, checking that inputs are correctly read,
    outputs are stored in the expected in-memory datasets, and the final
    data structures meet expectations (e.g. columns, non-emptiness).

Scope:
    - **Namespaced Node Execution**: Ensures each node (or node chain) is
      executed under the correct dataset namespace (e.g. "dataset1").
    - **Intermediate Outputs**: Confirms that outputs like "team_lexicon",
      "goals", and "vectorized_data" are written to `MemoryDataset` objects
      for each dataset (e.g. "dataset1.team_lexicon", "dataset2.goals", etc.).
    - **Chained Execution**: Validates that multi-node sequences
      (build_team_lexicon → get_goal_results → vectorize_data) function
      as an end-to-end process for each dataset.

Referenced Components:
    - `create_pipeline()`: Constructs a pipeline in which each dataset from
      `settings.DATASETS` has three nodes: `build_team_lexicon_node`,
      `get_goal_results_node`, and `vectorize_data_node`. Each node uses a
      namespace matching the dataset name, e.g. `"dataset1.build_team_lexicon_node"`.
    - **Nodes**:
        1. `build_team_lexicon`: Maps team names to unique indices.
        2. `get_goal_results`: Extracts and structures goal data.
        3. `vectorize_data`: Converts goal results into a final,
           ML-friendly DataFrame.

Testing Standards & Guidelines:
    1. **Integration Style**: Rather than testing each function in isolation,
       these tests run Kedro pipelines (or partial pipelines) with a
       `SequentialRunner` and in-memory data catalogs.
    2. **Namespaced Node Identification**: Nodes are selected via `.only_nodes()`
       with their fully qualified names (e.g. "dataset1.build_team_lexicon_node").
    3. **Catalog Handling**:
       - `test_catalog`: An in-memory `DataCatalog` with one input dataset
         and multiple output datasets per namespace (e.g.
         "dataset1.CSV" → "dataset1.goals").
       - If the node fails to produce valid output, the `MemoryDataset`
         remains empty, triggering assertions.
    4. **Edge Cases**: Although the focus is node-level integration, typical
       pipeline errors such as missing columns or empty data would still
       surface if the nodes raise exceptions or produce empty outputs.

Pre-requisites:
    - Kedro, PyTest, Pandas
    - A functioning Kedro project structure containing:
        - `settings.DATASETS` as a list of dataset names (e.g. ["dataset1", "dataset2"]).
        - `create_pipeline` from the `data_processing.pipeline` module.

How to Run:
    1. Install project dependencies (including Kedro, PyTest, Pandas).
    2. Place this file in your `tests/pipelines/data_processing/` directory.
    3. Execute `pytest -v tests/pipelines/data_processing/test_pipeline_etl.py`
       from the project root.

Expected Outcome:
    - Each test ensures that the relevant node(s) can be run within a
      namespaced pipeline, producing the correct in-memory outputs.
    - A successful run implies the data transformations are performed
      correctly for each dataset namespace (team lexicon, goals, vectorized
      data, etc.).
    - Failure indicates either an incorrectly configured pipeline, missing
      columns in the test data, or a mismatch between node input/outputs and
      the catalog dataset names.

===============================================================================
"""

from unittest.mock import patch

import pandas as pd
import pytest
from kedro.io import DataCatalog, MemoryDataset
from kedro.runner import SequentialRunner

from bundesliga import settings
from bundesliga.pipelines.data_processing.pipeline import create_pipeline


@pytest.fixture
def seq_runner():
    """
    Provides a fresh SequentialRunner for executing small Kedro pipelines.
    """
    return SequentialRunner()


@pytest.fixture
def sample_csv_data():
    """
    Returns minimal, valid match data as a DataFrame.
    Columns must match those expected by build_team_lexicon and get_goal_results.
    """
    data = {
        "HomeTeam": ["TeamA", "TeamB"],
        "AwayTeam": ["TeamC", "TeamA"],
        "FTHG": [2, 1],
        "FTAG": [1, 2],
    }
    return pd.DataFrame(data)


@pytest.fixture
def mock_settings_datasets():
    """
    Mocks or patches the settings.DATASETS list for controlled testing.
    In an actual project, you might define this fixture differently
    or skip it if you prefer real dataset names.
    """
    with patch.object(settings, "DATASETS", ["dataset1", "dataset2"]):
        yield


@pytest.fixture
def test_catalog(sample_csv_data):
    """
    Creates an in-memory DataCatalog. For each dataset in settings.DATASETS:
      - Adds '{dataset}.CSV' with sample_csv_data as input
      - Adds '{dataset}.team_lexicon', '{dataset}.goals', '{dataset}.vectorized_data'
        as blank MemoryDatasets for pipeline outputs.
    """
    catalog = DataCatalog()
    for ds in settings.DATASETS:
        catalog.add(f"{ds}.CSV", MemoryDataset(sample_csv_data))
        catalog.add(f"{ds}.team_lexicon", MemoryDataset())
        catalog.add(f"{ds}.goals", MemoryDataset())
        catalog.add(f"{ds}.vectorized_data", MemoryDataset())
    return catalog


class TestBuildTeamLexiconPipeline:
    def test_build_team_lexicon_node(
        self,
        seq_runner,
        mock_settings_datasets,
        test_catalog,
    ):
        """
        Verifies that the 'build_team_lexicon_node' executes correctly in each dataset namespace.

        This test runs a pipeline containing only the 'build_team_lexicon_node' for each dataset defined in settings.
        It checks that the output for each namespace (i.e., the 'team_lexicon' MemoryDataset) is produced, is not empty,
        and includes the required 'index' column.
        """
        datasets = settings.DATASETS
        node_names = [f"{ds}.build_team_lexicon_node" for ds in datasets]
        pipeline = create_pipeline().only_nodes(*node_names)

        # Execute pipeline and verify outputs
        seq_runner.run(pipeline, test_catalog)
        for ds in datasets:
            lexicon_key = f"{ds}.team_lexicon"
            lexicon_df = test_catalog.load(lexicon_key)
            assert lexicon_df is not None, f"No output written for '{lexicon_key}'."
            assert not lexicon_df.empty, f"'{lexicon_key}' is empty."
            assert "index" in lexicon_df.columns, f"Missing 'index' in '{lexicon_key}'."


class TestGetGoalResultsPipeline:
    def test_get_goal_results_node(
        self, seq_runner, mock_settings_datasets, test_catalog
    ):
        """
        Ensures that the 'get_goal_results_node' executes correctly when chained with the build_team_lexicon_node.

        This test runs a partial pipeline that includes both 'build_team_lexicon_node' and 'get_goal_results_node'
        for each dataset. It verifies that the 'goals' output is produced, is not empty, and contains the expected
        columns 'home_goals' and 'away_goals'.
        """
        datasets = settings.DATASETS
        build_nodes = [f"{ds}.build_team_lexicon_node" for ds in datasets]
        goal_nodes = [f"{ds}.get_goal_results_node" for ds in datasets]

        pipeline = create_pipeline().only_nodes(*(build_nodes + goal_nodes))

        seq_runner.run(pipeline, test_catalog)
        for ds in datasets:
            goals_key = f"{ds}.goals"
            goals_df = test_catalog.load(goals_key)
            assert goals_df is not None, f"'{goals_key}' not found."
            assert not goals_df.empty, f"'{goals_key}' is empty."
            for col in ("home_goals", "away_goals"):
                assert col in goals_df.columns, f"Missing '{col}' in '{goals_key}'."


class TestVectorizeDataPipeline:
    def test_vectorize_data_node(
        self, seq_runner, mock_settings_datasets, test_catalog
    ):
        """
        Validates that the complete data processing sub-pipeline (build_team_lexicon → get_goal_results → vectorize_data)
        produces a non-empty 'vectorized_data' output with the expected columns.

        This test runs the full sub-pipeline for each dataset namespace and checks that the 'vectorized_data' MemoryDataset
        contains the columns 'home_id', 'away_id', 'home_goals', 'away_goals', and 'toto', ensuring proper data transformation.
        """
        datasets = settings.DATASETS
        build_nodes = [f"{ds}.build_team_lexicon_node" for ds in datasets]
        goal_nodes = [f"{ds}.get_goal_results_node" for ds in datasets]
        vector_nodes = [f"{ds}.vectorize_data_node" for ds in datasets]

        pipeline = create_pipeline().only_nodes(
            *(build_nodes + goal_nodes + vector_nodes)
        )

        seq_runner.run(pipeline, test_catalog)
        for ds in datasets:
            vector_key = f"{ds}.vectorized_data"
            vector_df = test_catalog.load(vector_key)
            assert vector_df is not None, f"'{vector_key}' not written."
            assert not vector_df.empty, f"'{vector_key}' is empty."
            expected_cols = {"home_id", "away_id", "home_goals", "away_goals", "toto"}
            missing = expected_cols - set(vector_df.columns)
            assert not missing, f"Missing columns in '{vector_key}': {missing}"
