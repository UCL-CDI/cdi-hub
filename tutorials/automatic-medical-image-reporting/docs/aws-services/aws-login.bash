#!/bin/bash
set -Eeuxo pipefail

AWS_PROFILE="AWSAdministratorAccess-cdi-dev"
aws sso login --profile ${AWS_PROFILE}

