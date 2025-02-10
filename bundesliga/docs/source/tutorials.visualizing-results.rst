Visualizing results
===================

**Pipeline Graph with Kedro-Viz**

This Framework uses Kedro-Viz to visualize the pipeline graph. You can generate the graph by running the following command after you ran your pipeline:

.. code-block:: bash
    
    > kedro viz

This will generate a visualization of your pipeline in the browser at `http://localhost:4141 <http://localhost:4141>`_. For more information on Kedro-Viz, see the `Visualise the spaceflights project <https://docs.kedro.org/projects/kedro-viz/en/v6.6.1.post1/kedro-viz_visualisation.html>`_ or the `Kedro-Viz Demo <https://demo.kedro.org/>`_ for a quick overview of its functionalities.

**Experiment Tracking with MLflow**

The framework automatically logs parameters, metrics, and artifacts to MLflow during pipeline execution. You can use the MLflow UI to visualize and compare results from different runs.

.. code-block:: bash
    
    > mlflow ui


This will start the MLflow UI server, which you can access in your browser at `http://localhost:5000 <http://localhost:5000>`_. Each run initialized by the CLI will create a new run with the pipeline name as the run name. This run encapsulates all child runs, which are individual pipelines in the main pipeline. All matchpredictions are named as Soloruns, where aggregated modelresults are named as Aggregatedruns. For reference see :mod:`bundesliga.hooks.model_tracking_hooks`.



