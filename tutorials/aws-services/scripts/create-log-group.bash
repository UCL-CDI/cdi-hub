#!/bin/bash
set -Eeuxo pipefail

SCRIPT_PATH=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_PATH/../
MAINPATH=$PWD

# Set environment variables
export AWS_PROFILE="AWSAdministratorAccess-cdi-dev"
export AWS_REGION="eu-west-2"

aws logs create-log-group --log-group-name "/aws/batch/cdi-fargate-job" \
    --profile ${AWS_PROFILE} \
    --region ${AWS_REGION}
