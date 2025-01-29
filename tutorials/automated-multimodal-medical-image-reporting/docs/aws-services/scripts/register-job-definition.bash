#!/bin/bash
set -Eeuxo pipefail

# Create security group for Batch jobs
export AWS_PROFILE="AWSAdministratorAccess-cdi-dev"
export REPOSITORY_NAME="cdi-hub/awsbatch-demo"
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export AWS_REGION="eu-west-2"
PATH_TO_JOB_DEFINITION="../aws-batch/configs/job-definition.json" #TOTEST

aws batch register-job-definition \
    --job-definition-name "cdi-fargate-job-def" \
    --platform-capabilities FARGATE \
    --type container \
    --container-properties ${PATH_TO_JOB_DEFINITION} \ 
    --profile ${AWS_PROFILE} \
    --region ${AWS_REGION}
