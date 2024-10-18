
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