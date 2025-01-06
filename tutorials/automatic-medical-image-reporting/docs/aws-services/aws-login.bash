#!/bin/bash
set -Eeuxo pipefail

AWS_PROFILE=$1
aws sso login --profile ${AWS_PROFILE}

