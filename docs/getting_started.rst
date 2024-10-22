Getting Started
==============

Installation
-----------

Requirements
~~~~~~~~~~~

The following packages are required:

* ``PyTorch``: Deep learning framework
* ``Weights & Biases``: Experiment tracking
* ``Hydra``: Configuration management
* ``NumPy``: Numerical computing
* ``tatm``: Dataset loading support

You can install the core requirements using pip::

    pip install torch
    pip install wandb
    pip install hydra-core
    pip install numpy

For dataset loading, we currently use the ``tatm`` package also developed at the Kempner Institute. Install it using::

    pip install git+https://github.com/KempnerInstitute/tatm.git

More details about ``tatm`` can be found `here <https://github.com/KempnerInstitute/tatm/tree/dev>`_ 

.. note::
    In future versions, we will support default dataset loading without requiring external packages.


Installation Steps
~~~~~~~~~~~~~~~

1. Clone and install the package::

    git clone https://github.com/KempnerInstitute/tmrc.git
    cd tmrc
    pip install -e .

Running Experiments
~~~~~~~~~~~~~~~~

1. Login to Weights & Biases to enable experiment tracking::

    wandb login

2. Request compute resources. For example, to request an H100 GPU::

    salloc --partition=kempner_h100 --account=kempner_dev --ntasks=1 --cpus-per-task=8 --mem=8G --gres=gpu:1  --time=00-02:00:00

3. Launch training::

    python src/tmrc/tmrc_core/training/train.py

Configuration
^^^^^^^^^^^

By default, the training script uses the configuration defined in ``configs/default_train_config.yaml``. 

To use a custom configuration file::

    python src/tmrc/tmrc_core/training/train.py --config-name YOUR_CONFIG

.. note::
    The ``--config-name`` parameter should be specified without the ``.yaml`` extension.

.. tip::
    Configuration files should be placed in the ``configs/`` directory. For example, if your config is named ``my_experiment.yaml``, use ``--config-name my_experiment``