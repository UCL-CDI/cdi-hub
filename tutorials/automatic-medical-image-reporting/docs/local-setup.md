# Local setup 

## Install [uv](https://github.com/astral-sh/uv): "An extremely fast Python package manager".
```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Create venv
```
cd ~/cdi-hub/tutorials/automatic-medical-image-reporting
uv venv --python 3.12 # Create a virtual environment at .venv.
```

## Acvivate venv
```
source .venv/bin/activate #To activate the virtual environment:
deactivate
```

## Install python package deps
```
uv pip install --editable ".[test]" # Install the package in editable mode with test dependencies
uv pip install ."[learning]" # Install learning dependencies
```

## Clean up any existing installation and reinstall:
```
uv pip uninstall amir
uv pip install -e ".[test,learning]"
```

## Testing 
```
cd ~/cdi-hub/tutorials/automatic-medical-image-reporting
source .venv/bin/activate 
python src/amir/apis/data-preprocessing.py
```

## Pre-commit
```
pre-commit run -a
# SKIP=pylint pre-commit run --all-files
```
