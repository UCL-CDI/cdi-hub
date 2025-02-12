#!/bin/bash
set -Eeuxo pipefail

# Set environment variables
export AWS_PROFILE="AWSAdministratorAccess-cdi-dev"
export AWS_REGION="eu-west-2"
export JOB_ID="c6edfa7a-792f-4f6e-b35b-dbf4ff107a1e"

aws batch describe-jobs \
    --profile ${AWS_PROFILE} \
    --region ${AWS_REGION} \
    --job ${JOB_ID}

