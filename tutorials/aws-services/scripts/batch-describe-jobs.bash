#!/bin/bash
set -Eeuxo pipefail

SCRIPT_PATH=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_PATH/../
MAINPATH=$PWD

# Set environment variables
export AWS_PROFILE="AWSAdministratorAccess-cdi-dev"
export AWS_REGION="eu-west-2"

aws batch describe-jobs \
    --jobs <job-id> \
    --profile ${AWS_PROFILE} \
    --region ${AWS_REGION}
