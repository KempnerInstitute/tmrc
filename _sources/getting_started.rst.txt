Getting Started
===============

Installation
------------

Requirements
~~~~~~~~~~~~

The following packages are required:

* ``PyTorch``: Deep learning framework
* ``wandb``: Weights & Biases experiment tracking
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

More details about ``tatm`` can be found `here <https://github.com/KempnerInstitute/tatm/tree/dev>`_.

.. note::
    In future versions, we will support default dataset loading without requiring external packages.


Installation Steps
~~~~~~~~~~~~~~~~~~

1. If you are using the Kempner AI cluster, load required modules::

    module load python/3.12.5-fasrc01
    module load cuda/12.4.1-fasrc01
    module load cudnn/9.1.1.17_cuda12-fasrc01

If you are not using the Kempner cluster, install torch and cuda dependencies following instructions on the `PyTorch website <https://pytorch.org>`_.

TMRC has been tested with torch ``2.5.0+cu124``.

2. Create a Conda environment (if you are using the Kempner AI cluster, you may use ``mamba`` instead of ``conda``)::

    conda create -n tmrc_env python=3.10
    conda activate tmrc_env

3. Clone the repository::

    git clone git@github.com:KempnerInstitute/tmrc.git

4. Install the package in editable mode::

    cd tmrc
    pip install -e .

Running Experiments
~~~~~~~~~~~~~~~~~~~

1. Login to Weights & Biases to enable experiment tracking::

    wandb login

2. Request compute resources. For example, on the Kempner AI cluster, to request an H100 GPU::

    salloc --partition=kempner_h100 --account=<fairshare account> --ntasks=1 --cpus-per-task=24 --mem=375G --gres=gpu:1  --time=00-07:00:00

If you are not using the Kempner AI cluster, you can run experiments on your local machine (if you have a GPU) or on cloud services like AWS, GCP, or Azure.  TMRC should automatically find the available GPU.

3. Launch training::

    python src/tmrc/tmrc_core/training/train.py

Configuration
^^^^^^^^^^^^^

By default, the training script uses the configuration defined in ``configs/training/default_train_config.yaml``. 

To use a custom configuration file::

    python src/tmrc/tmrc_core/training/train.py --config-name YOUR_CONFIG

.. note::
    The ``--config-name`` parameter should be specified without the ``.yaml`` extension.

.. tip::
    Configuration files should be placed in the ``configs/training/`` directory. For example, if your config is named ``my_experiment.yaml``, use ``--config-name my_experiment``