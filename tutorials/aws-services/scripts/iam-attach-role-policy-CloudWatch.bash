#!/bin/bash
set -Eeuxo pipefail

SCRIPT_PATH=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_PATH/../
MAINPATH=$PWD

# Set environment variables
export AWS_PROFILE="AWSAdministratorAccess-cdi-dev"
export REPOSITORY_NAME="cdi-hub/awsbatch-demo"
export AWS_REGION="eu-west-2"
PATH_TO_JSON=file://${MAINPATH}/aws-batch/configs/trust-policy.json

aws iam attach-role-policy \
    --role-name ecsTaskExecutionRole \
    --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy
