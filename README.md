# bundesliga

[![Powered by Kedro](https://img.shields.io/badge/powered_by-kedro-ffc900?logo=kedro)](https://kedro.org)

## Overview

This is your new Kedro project with Kedro-Viz setup, which was generated using `kedro 0.19.10`.

Take a look at the [Kedro documentation](https://docs.kedro.org) to get started.

## How to install dependencies

First create your new conda environment and activate it. We use this to already install pymc dependencies.

```
conda create -c conda-forge -n your_env "pymc>=5"
conda activate your_env
```

To install the required dependencies, run:

```
pip install -r requirements.txt
```

## How to run the Kedro pipeline

First build neccessary files by running the etl pipeline:

```
kedro run --pipeline "etl"
```

After you can run either the whole pipeline with all engines defined in the settings.py (DYNAMIC_PIPELINES_MAPPING) or use the name to run it respectivly.

```
kedro run | kedro run --pipeline "pymc"
```

To define how many models and samples you want to draw go into parameters.py.

Options:
-   start_day = startingpoint for the model in the season to learn
-   walk_forward = how many models the pipeline should initiate and train
i.e start_day = 30 and walk_forward=2 will train and predict on match 30,31,32

## Visualize kedro pipeline and evaluations

To visualize the pipeline open kedro-viz in terminal. This will open your browser automatically.

```
kedro viz
```

To visualize the runs and logged metrics open the mlflow ui.

```
mlflow ui
```

located at http://localhost:5000/ after starting the ui in your terminal.

<!-- 
## How to test your Kedro project

Have a look at the files `src/tests/test_run.py` and `src/tests/pipelines/data_science/test_pipeline.py` for instructions on how to write your tests. Run the tests as follows:

```
pytest
```

To configure the coverage threshold, look at the `.coveragerc` file. -->
<!-- 
## Project dependencies

To see and update the dependency requirements for your project use `requirements.txt`. Install the project requirements with `pip install -r requirements.txt`.

[Further information about project dependencies](https://docs.kedro.org/en/stable/kedro_project_setup/dependencies.html#project-specific-dependencies)

## How to work with Kedro and notebooks

> Note: Using `kedro jupyter` or `kedro ipython` to run your notebook provides these variables in scope: `catalog`, `context`, `pipelines` and `session`.
>
> Jupyter, JupyterLab, and IPython are already included in the project requirements by default, so once you have run `pip install -r requirements.txt` you will not need to take any extra steps before you use them.

### Jupyter
To use Jupyter notebooks in your Kedro project, you need to install Jupyter:

```
pip install jupyter
```

After installing Jupyter, you can start a local notebook server:

```
kedro jupyter notebook
```

### JupyterLab
To use JupyterLab, you need to install it:

```
pip install jupyterlab
```

You can also start JupyterLab:

```
kedro jupyter lab
```

### IPython
And if you want to run an IPython session:

```
kedro ipython
```

### How to ignore notebook output cells in `git`
To automatically strip out all output cell contents before committing to `git`, you can use tools like [`nbstripout`](https://github.com/kynan/nbstripout). For example, you can add a hook in `.git/config` with `nbstripout --install`. This will run `nbstripout` before anything is committed to `git`.

> *Note:* Your output cells will be retained locally.

[Further information about using notebooks for experiments within Kedro projects](https://docs.kedro.org/en/develop/notebooks_and_ipython/kedro_and_notebooks.html).
## Package your Kedro project

[Further information about building project documentation and packaging your project](https://docs.kedro.org/en/stable/tutorial/package_a_project.html). -->
