#!/bin/bash
set -Eeuxo pipefail

# Set environment variables
export AWS_PROFILE="AWSAdministratorAccess-cdi-dev"
export AWS_REGION="eu-west-2"

aws batch create-job-queue \
    --job-queue-name "cdi-fargate-queue" \
    --state ENABLED \
    --priority 1 \
    --compute-environment-order order=1,computeEnvironment=cdi-fargate-env \
    --profile ${AWS_PROFILE} \
    --region ${AWS_REGION}
