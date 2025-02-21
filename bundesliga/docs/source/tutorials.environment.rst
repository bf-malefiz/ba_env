Setting up the Environment
==========================

Before you begin, ensure that you have all the necessary dependencies installed.
First create your new conda environment and activate it. We use this to install the necessary pymc dependencies.


.. code-block:: bash
    
    > conda create -c conda-forge -n your_env "pymc>=5"
    > conda activate your_env


To install the required dependencies, run:

.. code-block:: bash
    
    > pip install -r requirements.txt

Add mlflow default config file to your project.:

.. code-block:: bash
    
    > kedro mlflow init

For testing the environment or building docs make sure to install the project as a package:

.. code-block:: bash
    
    > pip install -e .
    > pytest
    
    > cd docs 
    > make html