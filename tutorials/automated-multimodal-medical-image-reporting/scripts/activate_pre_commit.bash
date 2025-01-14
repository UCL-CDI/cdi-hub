#!/bin/bash
set -Eeuxo pipefail

source .venv/bin/activate #To activate the virtual environment:
pre-commit run -a
