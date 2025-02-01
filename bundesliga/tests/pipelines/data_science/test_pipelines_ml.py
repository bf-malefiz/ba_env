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
    1. **Sub-Pipeline (Single Day)**:
       - `create_subpipeline_for_day(day)`: Asserts that day-based nodes
         (day_node, split_node, init_model_node, fit_node, predict_node,
         evaluate_node) exist and reference appropriate inputs/outputs.

    2. **Walk-Forward Pipeline**:
       - `create_walk_forward_pipeline(start_day, times_to_walk)`: Checks that
         all daily sub-pipelines in `[start_day..start_day+times_to_walk]` are
         created, yielding the correct total number of nodes.

    3. **ML Pipeline**:
       - `ml_pipeline(...)`: Builds on the walk-forward pipeline,
         verifying that day-based nodes are namespaced with the
         appropriate dataset + variant, and that the final pipeline has
         the expected node count.

    4. **Evaluation Pipeline**:
       - `eval_pipeline(...)`: Creates aggregator nodes to gather daily
         metrics from multiple engines and variants. Each aggregator node
         checks for daily metrics references.

    5. **Integration Test** (Optional):
       - Demonstrates a minimal run of the ML pipeline with mocks for
         model training/prediction, ensuring that a final metrics output
         (e.g. `metrics_1`) is successfully produced.

Referenced Components:
    - `create_subpipeline_for_day(day)`
    - `create_walk_forward_pipeline(start_day, times_to_walk)`
    - `ml_pipeline(...)`
    - `eval_pipeline(...)`
    - `create_model_definition_node(engine, variant, dataset_name)`

Testing Standards & Guidelines:
    1. **Node Verification**: Each sub-pipeline or aggregator pipeline must
       contain the correct nodes (with matching names).
    2. **I/O Checks**: For selected nodes, we verify that all required
       inputs/outputs (dataset names) match the pipeline’s design.
    3. **Walk-Forward Correctness**: We confirm the pipeline creates daily
       sub-pipelines for each day, with 6 nodes per day if following the
       pattern: (day_node → split → init → train → predict → evaluate).
    4. **Mocking for Integration**: In the final integration test, real
       model logic is mocked to avoid heavy computations or external
       dependencies, focusing purely on pipeline flow and final outputs.

Pre-requisites:
    - Python 3.x, Kedro, pytest, pandas
    - Functions and nodes:
        - `create_subpipeline_for_day`, `create_walk_forward_pipeline`,
          `ml_pipeline`, `eval_pipeline`, `create_model_definition_node`.
        - Node implementations: `split_time_data`, `init_model`, `train`,
          `predict_goals`, `evaluate`, etc. (these may be mocked).

Usage:
    1. Place this file under `tests/pipelines/data_science/`.
    2. Install required dependencies (Kedro, pytest, pandas).
    3. Run `pytest -v tests/pipelines/data_science/test_data_science_pipeline.py`.

Expected Outcome:
    - All tests pass if:
      (a) Each day’s sub-pipeline has 6 node references,
      (b) The aggregator nodes correctly reference daily metrics,
      (c) The pipeline produces a final `metrics_1` when run with mocked nodes,
      (d) Node naming and namespaces align with the configured dataset, variant,
          and day.

===============================================================================
"""
from unittest.mock import patch

import pandas as pd
import pytest
from bundesliga import settings
from bundesliga.pipelines.data_science.pipeline import (
    create_model_definition_node,
    create_subpipeline_for_day,
    create_walk_forward_pipeline,
    eval_pipeline,
    ml_pipeline,
)
from kedro.io import DataCatalog, MemoryDataset
from kedro.pipeline import Pipeline
from kedro.runner import SequentialRunner


@pytest.fixture
def day():
    """Sample day index for testing create_subpipeline_for_day."""
    return 10


@pytest.fixture
def start_day():
    """Sample start day for walk-forward pipeline tests."""
    return 5


@pytest.fixture
def times_to_walk():
    """Sample 'walk_forward' count for pipeline tests."""
    return 2


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
def seq_runner():
    """Provides a fresh SequentialRunner for pipeline execution."""
    return SequentialRunner()


@pytest.fixture
def catalog():
    """Provides an in-memory DataCatalog for integration-like tests."""
    return DataCatalog()


class TestCreateSubpipelineForDay:
    def test_subpipeline_nodes(self, day):
        """
        Checks that create_subpipeline_for_day(day) returns a pipeline with nodes:
          - day_node_{day}
          - split_node_{day}
          - init_model_node_{day}
          - fit_node_{day}
          - predict_node_{day}
          - evaluate_node_{day}

        and verifies key input/output references for day_node and evaluate_node.
        """
        sub_pipe = create_subpipeline_for_day(day)
        assert isinstance(
            sub_pipe, Pipeline
        ), "create_subpipeline_for_day should return a Kedro Pipeline."

        node_names = {n.name for n in sub_pipe.nodes}
        expected_names = {
            f"day_node_{day}",
            f"split_node_{day}",
            f"init_model_node_{day}",
            f"fit_node_{day}",
            f"predict_node_{day}",
            f"evaluate_node_{day}",
        }
        assert expected_names.issubset(
            node_names
        ), f"Missing expected node(s). Found: {node_names}"

        day_node = next(n for n in sub_pipe.nodes if n.name == f"day_node_{day}")
        assert not day_node.inputs, "day_node should have no inputs."
        assert day_node.outputs == [
            f"day_{day}"
        ], f"day_node_{day} must output 'day_{day}'."

        evaluate_node = next(
            n for n in sub_pipe.nodes if n.name == f"evaluate_node_{day}"
        )
        # Check that each expected dataset is in evaluate_node.inputs
        required_inputs = [
            f"model_{day}",
            f"day_{day}",
            f"test_data_{day}",
            f"predictions_{day}",
        ]
        for inp in required_inputs:
            assert (
                inp in evaluate_node.inputs
            ), f"{evaluate_node.name} missing input: '{inp}'"

        # (Optional) Check the private _inputs dict to confirm arg -> dataset
        # for arg_name, dataset_name in {
        #     "model": f"model_{day}",
        #     "day": f"day_{day}",
        # }.items():
        #     assert evaluate_node._inputs[arg_name] == dataset_name


class TestCreateWalkForwardPipeline:
    def test_walk_forward_pipeline_structure(self, start_day, times_to_walk):
        """
        Ensures create_walk_forward_pipeline(start_day, times_to_walk) produces
        sub-pipelines for each day in the range [start_day..start_day+times_to_walk].
        Each sub-pipeline has 6 nodes, so total: 6 * (times_to_walk+1).
        """
        wf_pipe = create_walk_forward_pipeline(start_day, times_to_walk)
        assert isinstance(wf_pipe, Pipeline)

        total_nodes = 6 * (times_to_walk + 1)
        assert (
            len(wf_pipe.nodes) == total_nodes
        ), f"Expected {total_nodes} nodes, got {len(wf_pipe.nodes)}."

        final_day = start_day + times_to_walk
        assert any(
            f"evaluate_node_{final_day}" == n.name for n in wf_pipe.nodes
        ), f"Missing final evaluate_node_{final_day}."


class TestMLPipeline:
    def test_ml_pipeline_namespaced_nodes(self):
        """
        Checks that ml_pipeline(...) creates a pipeline with
        6*(walk_forward+1) day-based nodes, each prefixed with
        'my_dataset.simple.' or similar naming convention.
        """
        start_day = 1
        walk_forward = 2
        engine = "some_engine"
        variant = "simple"
        dataset_name = "my_dataset"

        ml_pipe = ml_pipeline(start_day, walk_forward, engine, variant, dataset_name)
        assert isinstance(ml_pipe, Pipeline)

        node_names = [node.name for node in ml_pipe.nodes]
        expected_count = 6 * (walk_forward + 1)
        assert (
            len(node_names) == expected_count
        ), f"Expected {expected_count} nodes, found {len(node_names)}."

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

    def test_eval_pipeline_structure(self):
        """
        Checks that eval_pipeline(...) creates nodes for each
        engine/variant pair plus aggregator nodes referencing the day-based metrics.
        """
        startday = 10
        walks = 2
        # Example: one engine with two variants
        setting = [("pymc", ["simple", "toto"])]
        dataset_name = "my_dataset"

        ep = eval_pipeline(startday, walks, setting, dataset_name)
        assert isinstance(ep, Pipeline)

        nodes = ep.nodes
        # 2 modeldef + 2 aggregator => 4 total
        assert len(nodes) == 4, f"Expected 4 nodes, got {len(nodes)}."

        aggregator_nodes = [n for n in nodes if "aggregate_eval_metrics_node" in n.name]
        assert len(aggregator_nodes) == 2, "Should have 1 aggregator node per variant."

        for agg_node in aggregator_nodes:
            # aggregator name: e.g. "pymc.my_dataset.simple.aggregate_eval_metrics_node"
            variant_part = agg_node.name.split(".")[2]

            # aggregator_node.inputs is a list of dataset names
            inputs_list = agg_node.inputs
            assert any(
                "modeldefinitions" in ds for ds in inputs_list
            ), f"Aggregator '{agg_node.name}' missing modeldefinitions."

            for day_i in range(startday, startday + walks):
                # e.g. "pymc.my_dataset.simple.metrics_10"
                expected_metrics = f"pymc.my_dataset.{variant_part}.metrics_{day_i}"
                assert (
                    expected_metrics in inputs_list
                ), f"Aggregator '{agg_node.name}' missing day metrics '{expected_metrics}'."


class TestMLPipelineIntegration:
    def test_ml_pipeline_single_day_run(self, seq_runner, catalog, sample_data):
        """
        Minimal integration test for ml_pipeline with start_day=1, walk_forward=0.
        Mocks init/train/predict/evaluate to ensure pipeline flow produces
        'metrics_1' as final output.
        """
        with patch(
            "bundesliga.pipelines.data_science.pipeline.init_model",
            return_value="mock_model",
        ):
            with patch(
                "bundesliga.pipelines.data_science.pipeline.train",
                return_value="trained_model",
            ):
                with patch(
                    "bundesliga.pipelines.data_science.pipeline.predict_goals",
                    return_value="predictions",
                ):
                    with patch(
                        "bundesliga.pipelines.data_science.pipeline.evaluate",
                        return_value="some_metrics",
                    ):
                        # Setup in-memory data
                        catalog.add("team_lexicon", MemoryDataset("some_lexicon"))
                        catalog.add("vectorized_data", MemoryDataset(sample_data))
                        catalog.add(
                            "params:simple.model_options",
                            MemoryDataset({"any": "option"}),
                        )

                        # Build the pipeline with a single day
                        pipe = ml_pipeline(
                            start_day=1,
                            walk_forward=0,
                            engine="mock_engine",
                            variant="simple",
                            dataset_name="test_ds",
                        )

                        outputs = seq_runner.run(pipe, catalog)
                        metrics_key = "test_ds.simple.metrics_1"
                        assert (
                            metrics_key in outputs
                        ), f"'{metrics_key}' not produced by pipeline."
                        assert (
                            outputs[metrics_key] == "some_metrics"
                        ), f"Unexpected value in '{metrics_key}': {outputs[metrics_key]}"
