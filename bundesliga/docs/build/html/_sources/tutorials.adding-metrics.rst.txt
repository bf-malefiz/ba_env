Adding additional Metrics to MLflow
===================================
The framework provides a modular design that allows you to easily add new metrics to the pipeline. The steps below outline how to add a new metrics to the framework and integrate them easily into the existing pipeline. 

1. **Create a new metric**: 

Create a new function for your metric in :mod:`bundesliga.utils` and import it into the :mod:`bundesliga.pipelines.data_science.nodes_ml` module. 

2. **Add the metric to an existing evaluation function**:

Depending on your evluation function you need to add the metric to the corresponding function. You have 3 options to do so.

**`evaluate_match` function**: 
    This allows you to evaluate the metric for each match. The data you will get is match specific.
**`aggregate_dataset_metrics` function**: 
    This allows you to evaluate the metric for the whole dataset. The data you will get is dataset specific.
**`aggregate_model_metrics` function**: 
    This allows you to evaluate the metric for the whole model. The data you will get is model specific.


Now add your metric into the output dictionary of the corresponding function and your metric will be logged by MLflow automaticly.
