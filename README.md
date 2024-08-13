# tmrc

_Transformer model research codebase_


## Installation

- Step 1: Clone the repository
```bash
git clone git@github.com:KempnerInstitute/tmrc.git
```

- Step 2: Install the package in editable mode
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