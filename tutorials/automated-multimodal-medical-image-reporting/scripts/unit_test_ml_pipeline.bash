#!/bin/bash
set -Eeuxo pipefail

source .venv/bin/activate #To activate the virtual environment:
# pytest -vs tests/test_ml_pipeline.py::test_gpu_availability
# pytest -vs tests/test_ml_pipeline.py::test_CheXNet_CNN_Dataset
pytest -vs tests/test_ml_pipeline.py::test_train_eval_model
