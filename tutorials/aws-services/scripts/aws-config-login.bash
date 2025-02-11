#!/bin/bash
set -Eeuxo pipefail

# Set environment variables
export AWS_PROFILE="AWSAdministratorAccess-cdi-dev"
export AWS_REGION="eu-west-2"
export REPOSITORY_NAME="cdi-hub/awsbatch-demo"
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

aws sso login --profile ${AWS_PROFILE}

# Validate AWS Account ID
if [ -z "$AWS_ACCOUNT_ID" ]; then
    echo "Error: Failed to get AWS Account ID"
    exit 1
fi
echo "Using AWS Account: $AWS_ACCOUNT_ID"
echo "Account AWS Profile: $AWS_PROFILE"

aws configure list # Verify AWS CLI configuration


