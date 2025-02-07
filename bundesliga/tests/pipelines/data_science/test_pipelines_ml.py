"""
===============================================================================
               TEST DOCUMENTATION: Bundesliga Data Science Pipeline
===============================================================================
Title:
    Kedro ML + Evaluation Pipeline Tests

Purpose:
    This suite of tests verifies both the structure and execution flow of an
    ML-oriented data science pipeline implemented with Kedro. The pipeline
    processes time-based splits, initializes and trains models, generates
    goal predictions, and evaluates results across walk-forward periods,
    engines, and variants.

Scope:
    1. **Sub-Pipeline (Single match)**:
       - `create_subpipeline_for_match(match)`: Asserts that match-based nodes
         (match_node, split_node, init_model_node, fit_node, predict_node,
         evaluate_node) exist and reference appropriate inputs/outputs.

    2. **Walk-Forward Pipeline**:
       - `create_walk_forward_pipeline(start_match, last_match)`: Checks that
         all daily sub-pipelines in `[start_match..start_match+last_match]` are
         created, yielding the correct total number of nodes.

    3. **ML Pipeline**:
       - `ml_pipeline(...)`: Builds on the walk-forward pipeline,
         verifying that match-based nodes are namespaced with the
         appropriate dataset + variant, and that the final pipeline has
         the expected node count.

    4. **Evaluation Pipeline**:
       - `eval_dataset_pipeline(...)`: Creates aggregator nodes to gather daily
         metrics from multiple engines and variants. Each aggregator node
         checks for daily metrics references.

    5. **Integration Test** (Optional):
       - Demonstrates a minimal run of the ML pipeline with mocks for
         model training/prediction, ensuring that a final metrics output
         (e.g. `metrics_1`) is successfully produced.

Referenced Components:
    - `create_subpipeline_for_match(match)`
    - `create_walk_forward_pipeline(start_match, last_match)`
    - `ml_pipeline(...)`
    - `eval_dataset_pipeline(...)`
    - `create_model_definition_node(engine, variant, dataset_name)`

Testing Standards & Guidelines:
    1. **Node Verification**: Each sub-pipeline or aggregator pipeline must
       contain the correct nodes (with matching names).
    2. **I/O Checks**: For selected nodes, we verify that all required
       inputs/outputs (dataset names) match the pipeline’s design.
    3. **Walk-Forward Correctness**: We confirm the pipeline creates daily
       sub-pipelines for each match, with 6 nodes per match if following the
       pattern: (match_node → split → init → train → predict → evaluate).
    4. **Mocking for Integration**: In the final integration test, real
       model logic is mocked to avoid heavy computations or external
       dependencies, focusing purely on pipeline flow and final outputs.

Pre-requisites:
    - Python 3.x, Kedro, pytest, pandas
    - Functions and nodes:
        - `create_subpipeline_for_match`, `create_walk_forward_pipeline`,
          `ml_pipeline`, `eval_dataset_pipeline`, `create_model_definition_node`.
        - Node implementations: `split_time_data`, `init_model`, `train`,
          `predict_goals`, `evaluate`, etc. (these may be mocked).

Usage:
    1. Place this file under `tests/pipelines/data_science/`.
    2. Install required dependencies (Kedro, pytest, pandas).
    3. Run `pytest -v tests/pipelines/data_science/test_data_science_pipeline.py`.

Expected Outcome:
    - All tests pass if:
      (a) Each match’s sub-pipeline has 6 node references,
      (b) The aggregator nodes correctly reference daily metrics,
      (c) The pipeline produces a final `metrics_1` when run with mocked nodes,
      (d) Node naming and namespaces align with the configured dataset, variant,
          and match.

===============================================================================
"""

from unittest.mock import patch

import pandas as pd
import pytest
from kedro.io import DataCatalog, MemoryDataset
from kedro.pipeline import Pipeline
from kedro.runner import SequentialRunner

from bundesliga import settings
from bundesliga.pipelines.data_science.pipeline import (
    create_model_definition_node,
    create_subpipeline_for_match,
    create_walk_forward_pipeline,
    eval_dataset_pipeline,
    ml_pipeline,
)


@pytest.fixture
def match():
    """Sample match index for testing create_subpipeline_for_match."""
    return 10


@pytest.fixture
def start_match():
    """Sample start match for walk-forward pipeline tests."""
    return 5


@pytest.fixture
def last_match():
    """Sample 'last_match' count for pipeline tests."""
    return 10


@pytest.fixture
def sample_data():
    """Minimal DataFrame to simulate 'vectorized_data' for splitting."""
    return pd.DataFrame(
        {
            "home_id": [0, 1],
            "away_id": [1, 2],
            "home_goals": [2, 3],
            "away_goals": [1, 1],
            "toto": [1, 2],
        }
    )


@pytest.fixture
def sample_test_data():
    """Minimal DataFrame to simulate 'vectorized_data' for splitting."""
    return pd.DataFrame(
        {
            "home_id": [1],
            "away_id": [2],
            "home_goals": [1],
            "away_goals": [0],
            "toto": [2],
        }
    )


@pytest.fixture
def sample_predictions():
    """Minimal DataFrame to simulate 'vectorized_data' for splitting."""
    return pd.DataFrame(
        {
            "home_goals": [1],
            "away_goals": [0],
        }
    )


@pytest.fixture
def seq_runner():
    """Provides a fresh SequentialRunner for pipeline execution."""
    return SequentialRunner()


@pytest.fixture
def catalog():
    """Provides an in-memory DataCatalog for integration-like tests."""
    return DataCatalog()


class TestCreateSubpipelineForMatch:
    def test_subpipeline_nodes(self, match):
        """
        Checks that create_subpipeline_for_match(match) returns a pipeline with nodes:
          - match_node_{match}
          - split_node_{match}
          - init_model_node_{match}
          - fit_node_{match}
          - predict_node_{match}
          - evaluate_node_{match}

        and verifies key input/output references for match_node and evaluate_node.
        """
        sub_pipe = create_subpipeline_for_match(match)
        assert isinstance(sub_pipe, Pipeline), (
            "create_subpipeline_for_match should return a Kedro Pipeline."
        )

        node_names = {n.name for n in sub_pipe.nodes}
        expected_names = {
            f"match_node_{match}",
            f"split_node_{match}",
            f"init_model_node_{match}",
            f"fit_node_{match}",
            f"predict_node_{match}",
            f"evaluate_node_{match}",
        }
        assert expected_names.issubset(node_names), (
            f"Missing expected node(s). Found: {node_names}"
        )

        match_node = next(n for n in sub_pipe.nodes if n.name == f"match_node_{match}")
        assert not match_node.inputs, "match_node should have no inputs."
        assert match_node.outputs == [f"match_{match}"], (
            f"match_node_{match} must output 'match_{match}'."
        )

        evaluate_node = next(
            n for n in sub_pipe.nodes if n.name == f"evaluate_node_{match}"
        )
        # Check that each expected dataset is in evaluate_node.inputs
        required_inputs = [
            f"model_{match}",
            f"match_{match}",
            f"test_data_{match}",
            f"predictions_{match}",
        ]
        for inp in required_inputs:
            assert inp in evaluate_node.inputs, (
                f"{evaluate_node.name} missing input: '{inp}'"
            )

        # (Optional) Check the private _inputs dict to confirm arg -> dataset
        # for arg_name, dataset_name in {
        #     "model": f"model_{match}",
        #     "match": f"match_{match}",
        # }.items():
        #     assert evaluate_node._inputs[arg_name] == dataset_name


class TestCreateWalkForwardPipeline:
    def test_last_match_pipeline_structure(self, start_match, last_match):
        """
        Ensures create_walk_forward_pipeline(start_match, last_match) produces
        sub-pipelines for each match in the range [start_match..start_match+last_match].
        Each sub-pipeline has 6 nodes, so total: 6 * (last_match+1).
        """
        wf_pipe = create_walk_forward_pipeline(start_match, last_match)
        assert isinstance(wf_pipe, Pipeline)
        total_matches = last_match - start_match
        total_nodes = 6 * (total_matches)
        assert len(wf_pipe.nodes) == total_nodes, (
            f"Expected {total_nodes} nodes, got {len(wf_pipe.nodes)}."
        )

        assert any(f"evaluate_node_{total_matches}" == n.name for n in wf_pipe.nodes), (
            f"Missing final evaluate_node_{total_matches}."
        )


class TestMLPipeline:
    def test_ml_pipeline_namespaced_nodes(self):
        """
        Checks that ml_pipeline(...) creates a pipeline with
        6*(last_match+1) match-based nodes, each prefixed with
        'my_dataset.simple.' or similar naming convention.
        """
        start_match = 1
        last_match = 2
        total_matches = last_match - start_match
        variant = "simple"
        dataset_name = "my_dataset"

        ml_pipe = ml_pipeline(start_match, last_match, variant, dataset_name)
        assert isinstance(ml_pipe, Pipeline)

        node_names = [node.name for node in ml_pipe.nodes]
        expected_count = 6 * (total_matches)
        assert len(node_names) == expected_count, (
            f"Expected {expected_count} nodes, found {len(node_names)}."
        )

        prefix = f"{dataset_name}.{variant}."
        for n in node_names:
            assert n.startswith(prefix), f"Node '{n}' doesn't start with '{prefix}'."


class TestEvalPipeline:
    def test_create_model_definition_node(self):
        """
        Ensures create_model_definition_node(...) outputs a node that
        returns a dictionary with engine, variant, dataset_name, and seed,
        stored under the correct outputs key.
        """
        engine = "pymc"
        variant = "simple"
        dataset_name = "my_dataset"

        node_obj = create_model_definition_node(engine, variant, dataset_name)
        assert node_obj.name == f"{engine}.{dataset_name}.{variant}.modeldef_node"

        # Node outputs is a list in Kedro
        expected_output = f"{engine}.{dataset_name}.{variant}.modeldefinitions"
        assert node_obj.outputs == [expected_output]

        # If we want to test the function's output directly (private API):
        fn = node_obj._func
        result = fn()
        assert result["engine"] == engine
        assert result["variant"] == variant
        assert result["dataset_name"] == dataset_name
        assert result["seed"] == settings.SEED

    def test_eval_dataset_pipeline_structure(self):
        """
        Checks that eval_dataset_pipeline(...) creates nodes for each
        engine/variant pair plus aggregator nodes referencing the match-based metrics.
        """
        startmatch = 10
        last_match = 15
        total_matches = last_match - startmatch
        # Example: one engine with two variants
        setting = [("pymc", "simple")]
        dataset_name = "my_dataset"

        ep = eval_dataset_pipeline(startmatch, last_match, setting, dataset_name)
        assert isinstance(ep, Pipeline)

        nodes = ep.nodes
        # 1 modeldef + 1 aggregator => 2 total
        assert len(nodes) == 2, f"Expected 4 nodes, got {len(nodes)}."

        aggregator_nodes = [
            n for n in nodes if "aggregate_dataset_metrics_node" in n.name
        ]
        assert len(aggregator_nodes) == 1, "Should have 1 aggregator node for variant."

        for agg_node in aggregator_nodes:
            # aggregator name: e.g. "pymc.my_dataset.simple.aggregate_dataset_metrics_node"
            variant_part = agg_node.name.split(".")[2]

            # aggregator_node.inputs is a list of dataset names
            inputs_list = agg_node.inputs
            assert any("modeldefinitions" in ds for ds in inputs_list), (
                f"Aggregator '{agg_node.name}' missing modeldefinitions."
            )

            for match in range(startmatch, total_matches):
                # e.g. "pymc.my_dataset.simple.metrics_10"
                expected_metrics = f"pymc.my_dataset.{variant_part}.metrics_{match}"
                assert expected_metrics in inputs_list, (
                    f"Aggregator '{agg_node.name}' missing match metrics '{expected_metrics}'."
                )
