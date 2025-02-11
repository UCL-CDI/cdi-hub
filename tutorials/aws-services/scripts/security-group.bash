#!/bin/bash
set -Eeuxo pipefail

# Set environment variables
export AWS_PROFILE="AWSAdministratorAccess-cdi-dev"
export AWS_REGION="eu-west-2"
export VPC_ID="vpc-09501fd39230d7b0f"
aws ec2 create-security-group \
    --group-name "batch-compute-sg" \
    --description "Security group for AWS Batch compute environment" \
    --vpc-id ${VPC_ID} \
    --profile ${AWS_PROFILE} \
    --region ${AWS_REGION}
