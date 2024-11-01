
<p align="center">
  <a href="https://github.com/KempnerInstitute/tmrc/actions/workflows/deploy-docs.yml">
    <img src="https://github.com/KempnerInstitute/tmrc/actions/workflows/deploy-docs.yml/badge.svg?branch=develop" alt="docs">
  </a>
  <a href="https://github.com/KempnerInstitute/tmrc/actions/workflows/python-package.yml">
    <img src="https://github.com/KempnerInstitute/tmrc/actions/workflows/python-package.yml/badge.svg" alt="tests">
  </a>
  <a href="https://codecov.io/gh/KempnerInstitute/tmrc" > 
    <img src="https://codecov.io/gh/KempnerInstitute/tmrc/graph/badge.svg?token=PONKB6HEEH"/> 
  </a>
</p>


# TMRC

_Transformer model research codebase_

TMRC (Transformer Model Research Codebase) is a simple, explainable codebase to train transformer-based models. It was developed with simplicity and ease of modification in mind, particularly for researchers. The codebase will eventually be used to train foundation models and experiment with architectural and training modifications.  This code is currently still in development and currently supports single GPU use-cases; we plan to scale to multi-GPU and multimodal settings in the near future.

## Documentation
[TMRC Documentation](https://symmetrical-couscous-g63ee4k.pages.github.io/)


## Installation

- Step 1: Load required modules

  If you are using the Kempner AI cluster, load required modules:

  ```bash
  module load python/3.12.5-fasrc01
  module load cuda/12.4.1-fasrc01
  module load cudnn/9.1.1.17_cuda12-fasrc01 
  ```

  If you are not using the Kempner cluster, install torch and cuda dependencies following instructions on the [PyTorch website](https://pytorch.org). TMRC has been tested with torch `2.5.0+cu124` and Python `3.12`.

- Step 2: Create a Conda environment

  ```bash
  conda create -n tmrc_env python=3.12
  conda activate tmrc_env
  ```

- Step 3: Clone the repository

  ```bash
  git clone git@github.com:KempnerInstitute/tmrc.git
  ```

- Step 4: Install the package in editable mode

  ```bash
  cd tmrc
  pip install -e .
  ```

## Running Experiments

- Step 1: Login to Weights & Biases to enable experiment tracking

  ```bash
  wandb login
  ```

- Step 2: Request compute resources. For example, on the Kempner AI cluster, to request an H100 80GB GPU run

  ```bash
  salloc --partition=kempner_h100 --account=<fairshare account> --ntasks=1 --cpus-per-task=24 --mem=375G --gres=gpu:1  --time=00-07:00:00
  ```

  If you are not using the Kempner AI cluster, you can run experiments on your local machine (if you have a GPU) or on cloud services like AWS, GCP, or Azure.  TMRC should automatically find the available GPU.  If there are no GPUs available, it will run on CPU (though this is not recommended, since training will be prohibitively slow for any reasonable model size).

- Step 3: Activate the Conda environment

  ```bash
  conda activate tmrc_env
  ```

- Step 4: Launch training

  ```bash
  python src/tmrc/tmrc_core/training/train.py
  ```

### Configuration

By default, the training script uses the configuration defined in `configs/training/default_train_config.yaml`. 

To use a custom configuration file

    python src/tmrc/tmrc_core/training/train.py --config-name YOUR_CONFIG

> [!NOTE]
> The `--config-name` parameter should be specified without the `.yaml` extension.

> [!TIP]
> Configuration files should be placed in the `configs/training/` directory. For example, if your config is named `my_experiment.yaml`, use `--config-name my_experiment`

## Build the documentation locally

- Step 1: Install the required packages
  ```bash
  pip install -e '.[docs]'
  ```

- Step 2: Build the documentation
  ```bash
  cd docs
  make html
  ```

- Step 3: Open the documentation in your browser
  ```bash
  open _build/html/index.html
    ```