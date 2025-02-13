#!/bin/bash
set -Eeuxo pipefail

# Set environment variables
export AWS_PROFILE="AWSAdministratorAccess-cdi-dev"
export AWS_REGION="eu-west-2"

aws batch submit-job \
    --job-name batch-demo-$(date +%Y%m%d-%H%M%S) \
    --job-queue cdi-fargate-queue \
    --job-definition cdi-fargate-job-def \
    --profile ${AWS_PROFILE} \
    --region ${AWS_REGION}
