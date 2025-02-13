#!/bin/bash
set -Eeuxo pipefail

# Set environment variables
export AWS_PROFILE="AWSAdministratorAccess-cdi-dev"
export REPOSITORY_NAME="cdi-hub/awsbatch-demo"
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export AWS_REGION="eu-west-2"

# Delete job queue
aws batch update-job-queue \
    --job-queue cdi-fargate-queue \
    --state DISABLED \
    --profile ${AWS_PROFILE} \
    --region ${AWS_REGION}

aws batch delete-job-queue \
    --job-queue cdi-fargate-queue \
    --profile ${AWS_PROFILE} \
    --region ${AWS_REGION}

# Delete compute environment
aws batch update-compute-environment \
    --compute-environment cdi-fargate-env \
    --state DISABLED \
    --profile ${AWS_PROFILE} \
    --region ${AWS_REGION}

aws batch delete-compute-environment \
    --compute-environment cdi-fargate-env \
    --profile ${AWS_PROFILE} \
    --region ${AWS_REGION}

# Delete ECR repository
aws ecr delete-repository \
    --repository-name ${REPOSITORY_NAME} \
    --force \
    --profile ${AWS_PROFILE} \
    --region ${AWS_REGION}
