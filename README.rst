.. raw:: html

   <p align="center">
     <img src="docs/MAENVS4VRP banner 5.png" alt="MAEnvs4VRP Logo" width="650">
   </p>

MAEnvs4VRP is a library comprising multi-agent environments for simulating classic vehicle routing problems.

`Documentation <https://maenvs4vrp.readthedocs.io/en/latest/>`_ | `Install <#install>`_ | `Quickstart Notebook <https://maenvs4vrp.readthedocs.io/en/latest/content/start.html>`_ | `Train Your Model <#training>`_ | `Paper (IJOC) <https://pubsonline.informs.org/doi/10.1287/ijoc.2025.1211>`_ | `Preprint <https://arxiv.org/abs/2411.14411>`_

.. image:: https://colab.research.google.com/assets/colab-badge.svg
    :alt: Google Colab Badge
    :target: https://colab.research.google.com/github/MAEnvs4VRP/maenvs4vrp/blob/master/maenvs4vrp/learning_notebooks/1.0.0_quickstart_cvrptw.ipynb

What's NEW in v0.3!
=====================

- Added four new environments: **CVRP**, **HCVRP**, **PCVRP**, and **TOP**
- Introduced **Parallel Environments** — vectorized (batched) variants for five problems: CVRP, PCVRP, PCVRPTW, TOP, and TOPTW
- Added **Neuro Solver** baselines: Attention Model (REINFORCE, PPO, GRPO) and 2D-Ptr (REINFORCE)
- Sequential environments now support three **policy training modes** beyond agent-selection rules:

  - **Agent → Action**: agent is selected first, then an action is chosen for that agent
  - **Action → Agent**: an action is chosen first, then the agent to execute it is selected
  - **Joint (Agent, Action)**: agent and action are selected simultaneously as a pair

Environments
============

.. list-table:: Available Vehicle Routing Environments:
   :widths: 25 5 5
   :header-rows: 1

   * - Environment
     - Source
     - Docs
   * - CVRP (Capacitated Vehicle Routing Problem)
     - `Code <https://github.com/MAEnvs4VRP/maenvs4vrp/tree/master/maenvs4vrp/environments/cvrp>`_
     - `Docs <https://maenvs4vrp.readthedocs.io/en/latest/environments/cvrp/cvrp.html>`_
   * - CVRPSTW (Capacitated Vehicle Routing Problem with Soft Time Windows)
     - `Code <https://github.com/MAEnvs4VRP/maenvs4vrp/tree/master/maenvs4vrp/environments/cvrpstw>`_
     - `Docs <https://maenvs4vrp.readthedocs.io/en/latest/environments/cvrpstw/cvrpstw.html>`_
   * - CVRPTW (Capacitated Vehicle Routing Problem with Time Windows)
     - `Code <https://github.com/MAEnvs4VRP/maenvs4vrp/tree/master/maenvs4vrp/environments/cvrptw>`_
     - `Docs <https://maenvs4vrp.readthedocs.io/en/latest/environments/cvrptw/cvrptw.html>`_
   * - DSVRPTW (Dynamic Stochastic Vehicle Routing Problem with Time Windows)
     - `Code <https://github.com/MAEnvs4VRP/maenvs4vrp/tree/master/maenvs4vrp/environments/dsvrptw>`_
     - `Docs <https://maenvs4vrp.readthedocs.io/en/latest/environments/dsvrptw/dsvrptw.html>`_
   * - DVRPTW (Dynamic Vehicle Routing Problem with Time Windows)
     - `Code <https://github.com/MAEnvs4VRP/maenvs4vrp/tree/master/maenvs4vrp/environments/dvrptw>`_
     - `Docs <https://maenvs4vrp.readthedocs.io/en/latest/environments/dvrptw/dvrptw.html>`_
   * - GMTDVRP (General Multi-Tasking Depot Vehicle Routing Problems)
     - `Code <https://github.com/MAEnvs4VRP/maenvs4vrp/tree/master/maenvs4vrp/environments/gmtdvrp>`_
     - `Docs <https://maenvs4vrp.readthedocs.io/en/latest/environments/gmtdvrp/gmtdvrp.html>`_
   * - GMTVRP (General Multi-Tasking Vehicle Routing Problems)
     - `Code <https://github.com/MAEnvs4VRP/maenvs4vrp/tree/master/maenvs4vrp/environments/gmtvrp>`_
     - `Docs <https://maenvs4vrp.readthedocs.io/en/latest/environments/gmtvrp/gmtvrp.html>`_
   * - HCVRP (Heterogeneous Capacitated Vehicle Routing Problem)
     - `Code <https://github.com/MAEnvs4VRP/maenvs4vrp/tree/master/maenvs4vrp/environments/hcvrp>`_
     - `Docs <https://maenvs4vrp.readthedocs.io/en/latest/environments/hcvrp/hcvrp.html>`_
   * - MDVRPTW (Multi-Depot Vehicle Routing Problem with Time Windows)
     - `Code <https://github.com/MAEnvs4VRP/maenvs4vrp/tree/master/maenvs4vrp/environments/mdvrptw>`_
     - `Docs <https://maenvs4vrp.readthedocs.io/en/latest/environments/mdvrptw/mdvrptw.html>`_
   * - MTDVRP (Multi-Tasking Depot Vehicle Routing Problems)
     - `Code <https://github.com/MAEnvs4VRP/maenvs4vrp/tree/master/maenvs4vrp/environments/mtdvrp>`_
     - `Docs <https://maenvs4vrp.readthedocs.io/en/latest/environments/mtdvrp/mtdvrp.html>`_
   * - MTVRP (Multi-Tasking Vehicle Routing Problems)
     - `Code <https://github.com/MAEnvs4VRP/maenvs4vrp/tree/master/maenvs4vrp/environments/mtvrp>`_
     - `Docs <https://maenvs4vrp.readthedocs.io/en/latest/environments/mtvrp/mtvrp.html>`_
   * - PCVRP (Prize Collecting Vehicle Routing Problem)
     - `Code <https://github.com/MAEnvs4VRP/maenvs4vrp/tree/master/maenvs4vrp/environments/pcvrp>`_
     - `Docs <https://maenvs4vrp.readthedocs.io/en/latest/environments/pcvrp/pcvrp.html>`_
   * - PCVRPTW (Prize Collecting Vehicle Routing Problem with Time Windows)
     - `Code <https://github.com/MAEnvs4VRP/maenvs4vrp/tree/master/maenvs4vrp/environments/pcvrptw>`_
     - `Docs <https://maenvs4vrp.readthedocs.io/en/latest/environments/pcvrptw/pcvrptw.html>`_
   * - PDPTW (Pickup and Delivery Problem with Time Windows)
     - `Code <https://github.com/MAEnvs4VRP/maenvs4vrp/tree/master/maenvs4vrp/environments/pdptw>`_
     - `Docs <https://maenvs4vrp.readthedocs.io/en/latest/environments/pdptw/pdptw.html>`_
   * - SDVRPTW (Split Delivery Vehicle Routing Problem with Time Windows)
     - `Code <https://github.com/MAEnvs4VRP/maenvs4vrp/tree/master/maenvs4vrp/environments/sdvrptw>`_
     - `Docs <https://maenvs4vrp.readthedocs.io/en/latest/environments/sdvrptw/sdvrptw.html>`_
   * - TOP (Team Orienteering Problem)
     - `Code <https://github.com/MAEnvs4VRP/maenvs4vrp/tree/master/maenvs4vrp/environments/top>`_
     - `Docs <https://maenvs4vrp.readthedocs.io/en/latest/environments/top/top.html>`_
   * - TOPTW (Team Orienteering Problem with Time Windows)
     - `Code <https://github.com/MAEnvs4VRP/maenvs4vrp/tree/master/maenvs4vrp/environments/toptw>`_
     - `Docs <https://maenvs4vrp.readthedocs.io/en/latest/environments/toptw/toptw.html>`_

Parallel Environments
=====================

Vectorized (batched) environments that process multiple problem instances simultaneously, enabling faster training through parallelism.

.. list-table:: Available Parallel Environments:
   :widths: 25 5 5
   :header-rows: 1

   * - Environment
     - Source
     - Docs
   * - CVRP (Capacitated Vehicle Routing Problem)
     - `Code <https://github.com/MAEnvs4VRP/maenvs4vrp/tree/master/maenvs4vrp/parallel_environments/cvrp>`_
     - `Docs <https://maenvs4vrp.readthedocs.io/en/latest/parallel_environments/cvrp/cvrp.html>`_
   * - PCVRP (Prize Collecting Vehicle Routing Problem)
     - `Code <https://github.com/MAEnvs4VRP/maenvs4vrp/tree/master/maenvs4vrp/parallel_environments/pcvrp>`_
     - `Docs <https://maenvs4vrp.readthedocs.io/en/latest/parallel_environments/pcvrp/pcvrp.html>`_
   * - PCVRPTW (Prize Collecting Vehicle Routing Problem with Time Windows)
     - `Code <https://github.com/MAEnvs4VRP/maenvs4vrp/tree/master/maenvs4vrp/parallel_environments/pcvrptw>`_
     - `Docs <https://maenvs4vrp.readthedocs.io/en/latest/parallel_environments/pcvrptw/pcvrptw.html>`_
   * - TOP (Team Orienteering Problem)
     - `Code <https://github.com/MAEnvs4VRP/maenvs4vrp/tree/master/maenvs4vrp/parallel_environments/top>`_
     - `Docs <https://maenvs4vrp.readthedocs.io/en/latest/parallel_environments/top/top.html>`_
   * - TOPTW (Team Orienteering Problem with Time Windows)
     - `Code <https://github.com/MAEnvs4VRP/maenvs4vrp/tree/master/maenvs4vrp/parallel_environments/toptw>`_
     - `Docs <https://maenvs4vrp.readthedocs.io/en/latest/parallel_environments/toptw/toptw.html>`_

Neuro Solvers
=============

Built-in neural network baselines ready to train on any parallel environment.

.. list-table:: Available Neuro Solvers:
   :widths: 25 5 5
   :header-rows: 1

   * - Solver
     - Source
     - Docs
   * - Attention Model (REINFORCE / PPO / GRPO)
     - `Code <https://github.com/MAEnvs4VRP/maenvs4vrp/tree/master/maenvs4vrp/neuro_solvers/attention_model>`_
     - `Docs <https://maenvs4vrp.readthedocs.io/en/latest/neuro_solvers/attention_model/attention_model.html>`_
   * - 2D-Ptr (REINFORCE)
     - `Code <https://github.com/MAEnvs4VRP/maenvs4vrp/tree/master/maenvs4vrp/neuro_solvers/2d_ptr>`_
     - `Docs <https://maenvs4vrp.readthedocs.io/en/latest/neuro_solvers/two_d_ptr/two_d_ptr.html>`_

Install
==========

Prerequisites
-------------

The library requires Python 3.11 or higher for installation, and it has been tested and confirmed stable with Python 3.13.5.

We use `uv <https://docs.astral.sh/uv/>`_ for fast, reliable dependency management. If you're familiar with ``pip`` and ``virtualenv``, ``uv`` works similarly but is much faster and handles environment creation automatically.

Install uv:

.. code:: shell

    # macOS with Homebrew
    brew install uv

    # macOS and Linux (official installer)
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # Windows
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

    # Or with pip/pipx
    pip install uv

Installation
------------

Clone the repository and install the package:

.. code:: shell

    git clone https://github.com/MAEnvs4VRP/maenvs4vrp.git && cd maenvs4vrp
    uv sync

That's it! ``uv sync`` automatically:

- Creates a virtual environment in ``.venv/``
- Installs Python 3.13.5 (if needed)
- Installs all dependencies from ``pyproject.toml``
- Creates a ``uv.lock`` file for reproducible builds

.. note::
   The ``uv.lock`` file pins exact dependency versions for reproducibility.
   It's committed to the repository to ensure consistent environments across
   development and CI, but doesn't affect users who install the package via pip.

To run commands in this environment, use ``uv run``:

.. code:: shell

    uv run python your_script.py
    uv run pytest

Or activate the environment traditionally if you prefer:

.. code:: shell

    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    python your_script.py

**Development installation:**

By default, ``uv sync`` installs only the core runtime dependencies (numpy, torch, pandas, etc.).
To include optional dependencies for testing and documentation:

.. code:: shell

    uv sync --extra dev          # Add testing tools (pytest, jupyter)
    uv sync --extra docs         # Add documentation tools (sphinx, etc.)
    uv sync --extra dev --extra docs  # Add both
    uv sync --all-extras         # Add all optional dependencies

Working with uv
---------------

Here are the most common ``uv`` commands you'll need:

.. code:: shell

    # Sync/install dependencies
    uv sync                          # Install all dependencies from pyproject.toml
    uv sync --extra dev              # Include dev dependencies (pytest, etc.)
    uv sync --extra docs             # Include docs dependencies (sphinx, etc.)

    # Run commands in the environment
    uv run python script.py          # Run a Python script
    uv run pytest                    # Run tests
    uv run jupyter notebook          # Launch Jupyter

    # Manage dependencies (modifies pyproject.toml)
    uv add numpy                     # Add a new runtime dependency
    uv add --dev pytest              # Add a development dependency
    uv remove pandas                 # Remove a dependency

    # Update dependencies
    uv lock --upgrade                # Update lock file with latest versions
    uv sync                          # Apply the updated dependencies

    # Inspect environment
    uv pip list                      # List all installed packages
    uv pip show torch                # Show details for a specific package

    # Clean/reset environment
    rm -rf .venv                     # Delete the virtual environment
    uv sync                          # Recreate it fresh


For more details, see the `uv documentation <https://docs.astral.sh/uv/>`_.

Getting Started
===================

We've prepared five hands-on notebooks that walk you through the library's different functionalities and environments. Feel free to explore them and use them as a starting point for your own experiments.

.. list-table::
   :widths: 20 30 50
   :header-rows: 1

   * - Notebook
     - Description
     - Colab
   * - `01: Quickstart <https://maenvs4vrp.readthedocs.io/en/latest/learning_notebooks/1.0.0_quickstart_cvrptw.html>`_
     - Learning MAEnvs4VRP basic usage.
     - |colab-quickstart|
   * - `02: MAEnvs4VRP library <https://maenvs4vrp.readthedocs.io/en/latest/learning_notebooks/2.0.0_maenvs4vrp_exploration_and_challenges.html>`_
     - Exploring MAEnvs4VRP library with challenges.
     - |colab-challenges|
   * - `03: Multi-Tasking Environments <https://maenvs4vrp.readthedocs.io/en/latest/learning_notebooks/3.0.0_multitask_environments.html>`_
     - Exploring MAEnvs4VRP multi-tasking environments.
     - |colab-multitask|
   * - `04: Stochastic Environments <https://maenvs4vrp.readthedocs.io/en/latest/learning_notebooks/4.0.0_maenvs4vrp_stochastic_environments.html>`_
     - Adapting MAEnvs4VRP deterministic environments into stochastic ones.
     - |colab-stochastic|
   * - `05: PyVRP <https://maenvs4vrp.readthedocs.io/en/latest/learning_notebooks/5.0.0_PyVRP_cvrptw_solver.html>`_
     - Exploring PyVRP on MAEnvs4VRP instances and environments.
     - |colab-PyVRP|

.. |colab-quickstart| image:: https://colab.research.google.com/assets/colab-badge.svg
   :alt: Google Colab Badge
   :target: https://colab.research.google.com/github/MAEnvs4VRP/maenvs4vrp/blob/master/maenvs4vrp/learning_notebooks/1.0.0_quickstart_cvrptw.ipynb
.. |colab-challenges| image:: https://colab.research.google.com/assets/colab-badge.svg
   :alt: Google Colab Badge
   :target: https://colab.research.google.com/github/MAEnvs4VRP/maenvs4vrp/blob/master/maenvs4vrp/learning_notebooks/2.0.0_maenvs4vrp_exploration_and_challenges.ipynb
.. |colab-multitask| image:: https://colab.research.google.com/assets/colab-badge.svg
   :alt: Google Colab Badge
   :target: https://colab.research.google.com/github/MAEnvs4VRP/maenvs4vrp/blob/master/maenvs4vrp/learning_notebooks/3.0.0_multitask_environments.ipynb
.. |colab-stochastic| image:: https://colab.research.google.com/assets/colab-badge.svg
   :alt: Google Colab Badge
   :target: https://colab.research.google.com/github/MAEnvs4VRP/maenvs4vrp/blob/master/maenvs4vrp/learning_notebooks/4.0.0_maenvs4vrp_stochastic_environments.ipynb
.. |colab-PyVRP| image:: https://colab.research.google.com/assets/colab-badge.svg
   :alt: Google Colab Badge
   :target: https://colab.research.google.com/github/MAEnvs4VRP/maenvs4vrp/blob/master/maenvs4vrp/learning_notebooks/5.0.0_PyVRP_cvrptw_solver.ipynb

Training
=============

Two baseline models are available, which can be trained with:

.. code-block:: python

    uv run python maenvs4vrp/learning/mardam/train_mardam.py --vrp_env toptw --num_agents 5 --num_nodes 51  --val_set servs_50_agents_5 --selection stime

.. code-block:: python

    uv run python maenvs4vrp/learning/madyam/train_madyam.py --vrp_env toptw --num_agents 5 --num_nodes 51  --val_set servs_50_agents_5 --selection stime

Unit Testing
=================

Unit tests are located in the `/tests/unit/` directory.

First, ensure you have the development dependencies installed:

.. code-block:: bash

    uv sync --extra dev

Then you can run individual tests as follows:

.. code-block:: bash

    uv run pytest tests/unit/test_environment_setup.py

.. code-block:: bash

    uv run pytest tests/unit/environments/seed_test.py

To run the full unit test suite and verify compatibility across different environments with customizable parameters, use:

.. code-block:: bash

    uv run pytest --device cpu --batch 1 --num_agents 2 --num_nodes 11

For additional details and examples, please refer to the documentation.


Directory Tree Structure
===========================

.. code:: text

    ├───maenvs4vrp
    │   ├───core
    │   ├───environments
    │   │   ├───cvrp
    │   │   ├───cvrpstw
    │   │   ├───cvrptw
    │   │   ├───dsvrptw
    │   │   ├───dvrptw
    │   │   ├───gmtdvrp
    │   │   ├───gmtvrp
    │   │   ├───hcvrp
    │   │   ├───mdvrptw
    │   │   ├───mtdvrp
    │   │   ├───mtvrp
    │   │   ├───pcvrp
    │   │   ├───pcvrptw
    │   │   ├───pdptw
    │   │   ├───sdvrptw
    │   │   ├───top
    │   │   ├───toptw
    │   ├───neuro_solvers
    │   │   ├───attention_model
    │   │   ├───two_d_ptr
    │   ├───parallel_environments
    │   │   ├───cvrp
    │   │   ├───pcvrp
    │   │   ├───pcvrptw
    │   │   ├───top
    │   │   ├───toptw
    │   ├───learning_notebooks
    │   ├───utils
    ├───tests
    │   ├───unit
    │   │   ├───environments

Citation
===============

To credit the library in your publications, use this citation:

.. code-block:: bibtex

    @article{gama2026maenvs4vrp,
      title={Multi-Agent Environments for Vehicle Routing Problems},
      author={Ricardo Gama and Ricardo Cunha and Daniel Fuertes and Carlos R. del-Blanco and Hugo L. Fernandes},
      year={2026},
      journal={INFORMS Journal on Computing},
      doi={10.1287/ijoc.2025.1211},
      url={https://pubsonline.informs.org/doi/10.1287/ijoc.2025.1211},
      note={\url{https://github.com/MAEnvs4VRP/maenvs4vrp}}
    }

Contributing
============
We welcome contributions to **MAEnvs4VRP**!
If you'd like to use this library in your academic research/industry projects, or if you have suggestions, feature requests, or any feedback, we'd be happy to hear from you.

Feel free to `open an issue <https://github.com/MAEnvs4VRP/maenvs4vrp/issues>`_ or submit a `pull request <https://github.com/MAEnvs4VRP/maenvs4vrp/pulls>`_. If you would like to contribute, please check out our contribution guidelines `here <https://github.com/MAEnvs4VRP/maenvs4vrp/blob/pre_commit_setup/.github/CONTRIBUTING.rst>`_. We welcome and look forward to all contributions to MAEnvs4vrp


Acknowledgements
=================
MAEnvs4VRP has been inspired by, and benefits from, the ideas and tooling of the broader open-source community. In particular, we would like to thank `PettingZoo <https://www.pettingzoo.ml/>`_,
`Flatland <https://github.com/flatland-association/flatland-rl/>`_, `MARDAM <https://gitlab.inria.fr/gbono/mardam>`_, `RL4CO <https://rl4co.readthedocs.io/en/latest//>`_, `RoutFinder <https://github.com/ai4co/routefinder/tree/main//>`_, `PyVRP <https://pyvrp.org//>`_ .
