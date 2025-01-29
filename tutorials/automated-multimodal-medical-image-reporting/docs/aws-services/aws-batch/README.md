# Workflow to setup AWS Batch
## Table of Contents
- [Prerequisites](#prerequisites)
- [Architecture Overview](#architecture-overview)
- [Initial Setup](#initial-setup)
- [Container Development](#container-development)
- [AWS Infrastructure Setup](#aws-infrastructure-setup)
- [Job Management](#job-management)
- [Monitoring and Troubleshooting](#monitoring-and-troubleshooting)
- [Cleanup](#cleanup)

## Prerequisites
- AWS CLI installed and configured
- Docker installed locally
- Access to UCL cloud portal
- Required AWS profiles:
  - `cdi-innov-dev` or `arc-playpen-collaborations`
  - `AWSAdministratorAccess-cdi-dev`

## Architecture Overview
This setup uses:
- AWS Batch for job orchestration
- AWS Fargate for serverless compute
- ECR for container image storage
- CloudWatch for logging

## Initial Setup

### Log in into your AWS access portal
Go to https://ucl-cloud.awsapps.com/start and select either `cdi-innov-dev` or `arc-playpen-collaborations` to launch AWSAdministratorAccess

### Log in with your AWS profile
* Log in
```bash
bash ../scripts/aws-config-login.bash
```

### Create ECR Repository [:link:](https://eu-west-2.console.aws.amazon.com/batch/home?region=eu-west-2#job-definition/ec2/new) 
```bash
bash ../scripts/create-repository.bash
```
Generating [registerjob.yaml](configs/registerjob.yaml)
See [aws-console](https://eu-west-2.console.aws.amazon.com/ecr/private-registry/repositories?region=eu-west-2).

## Container development
### Build docker image
Please refer to [Dockerfile](Dockerfile) and [docker-compose](docker-compose.yml)
```bash
docker compose -f docker-compose.yml build #Building estimated time
```
### Verify image size
```
docker images
#REPOSITORY   TAG               IMAGE ID       CREATED         SIZE
#awsbatch     latest            811d5b0b07f9   2 minutes ago   452MB
```

### Local container testing
Before deploying to AWS Batch, test your container locally:
```bash
bash ../scripts/launch_and_test_docker_image_locally.bash
```
List all the containers available locally  (incl the size)
```bash
docker ps -as
```

### Stop container and remove it
```bash
bash ../scripts/stop_container_and_removeit.bash
```

### Push to ECR [:link:](https://docs.aws.amazon.com/AmazonECR/latest/userguide/docker-push-ecr-image.html)
See [aws-settings.yaml](../configs/aws-settings.yaml).
```bash
bash ../scripts/push_image_to_ecr.bash
```
TODO: See [aws-console](?) 


## AWS Infrastructure Setup
### VPC Requirements
Before creating the compute environment, ensure you have:
- A VPC with at least one subnet
- For internet access, either:
  - Private subnets with NAT Gateway
  - Public subnets with Internet Gateway
- Security groups configured for your workload

* Get your VPC ID and subnet IDs
```bash
bash ../scripts/vpcs-subnets.bash 
```
Generates [vpcs-subnets.yaml](configs/vpcs-subnets.yaml)

* Create security group for Batch jobs. Ensure you use your vpc is is in [security-group.bash](../scripts/security-group.bash)
```bash
bash ../scripts/security-group.bash
```    

### Create Compute Environment [:link:](https://aws.amazon.com/blogs/aws/run-large-scale-simulations-with-aws-batch-multi-container-jobs/)
* Ensure you use your Subnets and Security Group Ids in [create-compute-env.bash](../scripts/create-compute-env.bash)
```bash
bash ../scripts/create-compute-env.bash
```

### Create Job Queue
```bash
bash ../scripts/create-job-queue.bash
```

### Create Job Definition
* Generate the final JSON with resolved variables using [`job-definition.template.json`](configs/job-definition.template.json)
```bash 
bash ../scripts/resolve-variables-for-template.bash
```

* Register the job definition [:link:](https://docs.aws.amazon.com/batch/latest/userguide/when-to-use-fargate.html)
```bash
bash ../scripts/register-job-definition.bash
```

:nut_and_bolt: :nut_and_bolt: :nut_and_bolt: :nut_and_bolt:

## Job Management 
### Submit a Job
```bash
aws batch submit-job \
    --job-name batch-demo-$(date +%Y%m%d-%H%M%S) \
    --job-queue cdi-fargate-queue \
    --job-definition cdi-fargate-job-def \
    --profile ${AWS_PROFILE} \
    --region ${AWS_REGION}
```
### Monitor Job Status
```bash
# Get job status
aws batch describe-jobs \
    --jobs <job-id> \
    --profile ${AWS_PROFILE} \
    --region ${AWS_REGION}

# List all jobs in queue
aws batch list-jobs \
    --job-queue cdi-fargate-queue \
    --profile ${AWS_PROFILE} \
    --region ${AWS_REGION}
```
## Monitoring and Troubleshooting
### View CloudWatch Logs
```bash
aws logs get-log-events \
    --log-group-name /aws/batch/job \
    --log-stream-name <job-stream-name> \
    --profile ${AWS_PROFILE} \
    --region ${AWS_REGION}
```
### Common Issues
- Check IAM roles and permissions
- Verify network configuration
- Confirm container image accessibility
- Review resource limits
- Check CloudWatch logs for errors

## Cleanup
### Remove Resources
```bash
bash ../scripts/cleanup.bash
```
