.. BA - Bayesian Model Evaluation documentation master file, created by
   sphinx-quickstart on Fri Feb  7 22:21:23 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

BA - Bayesian Model Evaluation Documentation
============================================

Welcome to the documentation for the Bayesian model evaluation framework, built using Kedro and MLflow. This framework provides tools and pipelines for evaluating Bayesian models in a robust and reproducible way.

Key Features
------------

* **Kedro Integration**: Leverages Kedro's pipeline structure for organizing and managing model training and evaluation workflows.
* **MLflow Tracking**: Integrates with MLflow to track experiments, log parameters, metrics, and artifacts, facilitating model comparison and reproducibility.
* **Modular Design**: Provides a flexible and extensible framework for incorporating various Bayesian models and evaluation metrics.
* **Data Validation**: Includes data validation checks to ensure data integrity and consistency throughout the pipeline.
* **Comprehensive Metrics**: Offers a wide range of evaluation metrics and incorperates easily extensible custom metrics which will log in MLflow automaticly.

Getting Started
---------------

This documentation provides detailed information on how to use the framework, including:

* Setting up the project environment and installing dependencies.
* Defining and running Kedro pipelines for model training and evaluation.
* Logging and visualizing results with MLflow.
* Extending the framework with custom models and metrics.

Documentation Structure
-----------------------

The documentation is organized as follows:

* **Modules**: Contains detailed API documentation for all modules and classes in the framework.
* **Tutorials**: Provides step-by-step guides on how to use the framework for specific use cases.
* **Examples**: Includes example projects demonstrating the framework's capabilities.

Contributing
------------

Contributions to the framework and documentation are welcome! Please see the contribution guidelines for more information.

License
-------
This project is licensed under the MIT License. All code and documentation is free to use,
modify, and distribute as long as proper attribution is provided.

.. toctree::
   bundesliga
   tutorials
   license
   :maxdepth: 4
   :caption: Contents:

