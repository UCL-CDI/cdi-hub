# Local setup 

## Setting up GPU drivers
* See [NVIDIA CUDA Installation Guide for Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/).
* Current local hardware capbilities
```
CUDA Version: 12.6
Driver Version: 560.28.03      
NVIDIA RTX A2000 8GB Laptop
```

## Install [uv](https://github.com/astral-sh/uv): "An extremely fast Python package manager".
```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Create venv
```
cd ~/cdi-hub/tutorials/automated-multimodal-medical-image-reporting
uv venv --python 3.12 # Create a virtual environment at .venv.
```

## Acvivate venv
```
source .venv/bin/activate #To activate the virtual environment:
deactivate
```

## Install python package deps
```
uv pip install --editable ".[test, learning]" # Install the package in editable mode with test and learning dependencies
uv pip install ."[learning]" # Install learning dependencies
```

## Clean up any existing installation and reinstall:
```
uv pip uninstall amir
uv pip install -e ".[test,learning]"
```

## Testing 
```
cd ~/cdi-hub/tutorials/automated-multimodal-medical-image-reporting
source .venv/bin/activate 
pytest -vs tests/test_ml_pipeline.py::test_CheXNet_CNN_Dataset
pytest -vs tests/test_ml_pipeline.py::test_train_eval_model
```

## Pre-commit
```
pre-commit run -a
# SKIP=pylint pre-commit run --all-files
```
