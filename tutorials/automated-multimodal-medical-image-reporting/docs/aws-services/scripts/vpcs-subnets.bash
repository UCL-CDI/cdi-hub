#!/bin/bash
set -Eeuxo pipefail

# Set environment variables
export AWS_PROFILE="AWSAdministratorAccess-cdi-dev"
export AWS_REGION="eu-west-2"
aws ec2 describe-vpcs --profile ${AWS_PROFILE} --region ${AWS_REGION}
aws ec2 describe-subnets --profile ${AWS_PROFILE} --region ${AWS_REGION}

