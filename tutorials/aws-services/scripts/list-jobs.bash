#!/bin/bash
set -Eeuxo pipefail

# Set environment variables
export AWS_PROFILE="AWSAdministratorAccess-cdi-dev"
export AWS_REGION="eu-west-2"

aws batch list-jobs \
    --job-queue cdi-fargate-queue \
    --profile ${AWS_PROFILE} \
    --region ${AWS_REGION}
