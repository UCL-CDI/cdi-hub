#!/bin/bash
set -Eeuxo pipefail

# Set environment variables
export AWS_PROFILE="AWSAdministratorAccess-cdi-dev"
export AWS_REGION="eu-west-2"
export BATCH_JOB="/aws/batch/job"
export STREAM_NAME="<job-stream-name>"

aws logs get-log-events \
    --log-group-name ${BATCH_JOB} \
    --log-stream-name ${STREAM_NAME} \
    --profile ${AWS_PROFILE} \
    --region ${AWS_REGION}
