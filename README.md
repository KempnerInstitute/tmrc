
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


# tmrc

_Transformer model research codebase_


## Installation 


- Step 1: Load required modules

```bash
module load python/3.12.5-fasrc01
module load cuda/12.4.1-fasrc01
module load cudnn/9.1.1.17_cuda12-fasrc01 
```

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

    wandb login

- Step 2: Request compute resources. For example, to request an H100 GPU

    salloc --partition=kempner_h100 --account=<your FASRC account> --ntasks=1 --cpus-per-task=8 --mem=8G --gres=gpu:1  --time=00-02:00:00

- Step 3: Launch training

    python src/tmrc/tmrc_core/training/train.py

### Configuration

By default, the training script uses the configuration defined in ``configs/default_train_config.yaml``. 

To use a custom configuration file

    python src/tmrc/tmrc_core/training/train.py --config-name YOUR_CONFIG

[!NOTE]
The ``--config-name`` parameter should be specified without the ``.yaml`` extension.

[!TIP]
Configuration files should be placed in the ``configs/`` directory. For example, if your config is named ``my_experiment.yaml``, use ``--config-name my_experiment``

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