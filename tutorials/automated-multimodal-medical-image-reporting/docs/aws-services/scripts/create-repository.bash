export AWS_PROFILE="AWSAdministratorAccess-cdi-dev"
export AWS_REGION="eu-west-2"
export REPOSITORY_NAME="cdi-hub/awsbatch-demo"

aws ecr create-repository \
    --repository-name ${REPOSITORY_NAME} \
    --region ${AWS_REGION} \
    --profile ${AWS_PROFILE}
