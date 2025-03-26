#!/bin/bash
set -Eeuxo pipefail

# Set environment variables
export AWS_PROFILE="AWSAdministratorAccess-cdi-dev"
export AWS_REGION="eu-west-2"
export SECURITY_GROUPID="sg-0b06fbe2a4f1e1cf2"
export SUBNETS_ID="subnet-0923a8554ca557417"
#"SubnetId": "subnet-0668db1341170459c"
#"SubnetId": "subnet-0708a2d7619ba4d48"

aws batch update-compute-environment \
	--compute-environment cdi-fargate-env \
	--state ENABLED \
	--profile ${AWS_PROFILE} \
	--region ${AWS_REGION}
