#!/bin/bash
set -Eeuxo pipefail

AWS_PROFILE="${1:-AWSAdministratorAccess-cdi-dev}"
echo $AWS_PROFILE

aws sso login --profile ${AWS_PROFILE}

