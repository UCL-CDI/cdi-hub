#!/bin/bash
set -Eeuxo pipefail

# Set environment variables
export AWS_PROFILE="AWSAdministratorAccess-cdi-dev"
export AWS_REGION="eu-west-2"

aws batch update-job-queue \
    --job-queue "arn:aws:batch:eu-west-2:975050006673:job-queue/cdi-fargate-queue" \
    --region ${AWS_REGION} \
    --state ENABLED
