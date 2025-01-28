# AWS Batch

## Log in into your AWS access portal
Go to https://ucl-cloud.awsapps.com/start and select either `cdi-innov-dev` or `arc-playpen-collaborations` to launch AWSAdministratorAccess

## Log in with your AWS profile
* Log in
```
bash ../scripts/aws-login.bash 
```

## Settting up Elastic Compute Cloud (Amazon EC2) orchestration [:link:](https://docs.aws.amazon.com/batch/latest/userguide/getting-started-ec2.html) [:link:](https://eu-west-2.console.aws.amazon.com/batch/home?region=eu-west-2#wizard)
1. Configure job and orchestration   
    #Enable using Spot instances[can be interrupted with a two minute notification when EC2]
    #VPC [aws-controltower-vpc]; 
2. Create compute environment. Name: ammir-ec2-comp-env   
3. Create a job queue. Name:  getting-started-ec2-job-queue-ammir   
    #setup security group [BatchEnvironmentDefaultSG]; 
4. Create a job definition. Name: getting-started-ec2-job-definition-ammir   
    Container: public.ecr.aws/amazonlinux/amazonlinux:latest    
    Container configuration; No commands
5. Create a job. Name: getting-started-ec2-job-ammir    
6. Review and create   
7. Confirmation of creation    
    Compute environment: ammir-ec2-comp-env 
    Job queue: getting-started-ec2-job-queue-ammir 
    Job definition: getting-started-ec2-job-definition-ammir 
    Job: getting-started-ec2-job-ammir 

## setup docker images
### Build
```
docker compose -f docker-compose.yml build #Building estimated time

# docker images
#REPOSITORY   TAG               IMAGE ID       CREATED         SIZE
#awsbatch     latest            811d5b0b07f9   2 minutes ago   452MB
```

### Stop container and remove it
```
bash ../scripts/stop_container_and_removeit.bash
```

## aws
* login
```
AWS_PROFILE=AWSAdministratorAccess-cdi-dev
bash ../scripts/aws-login.bash ${AWS_PROFILE}
#aws sso logout 
```


## Config settings
See [aws-settings.yaml](../configs/aws-settings.yaml).
```
AWS_PROFILE="AWSAdministratorAccess-cdi-dev"
REPOSITORY_NAME="cdi-hub/test-container"
#REPOSITORY_NAME="cdi-hub/aws-samples"
AWS_REGION="eu-west-2"
```

### Create job definition [:link:](https://eu-west-2.console.aws.amazon.com/batch/home?region=eu-west-2#job-definition/ec2/new) 
Generating [registerjob.yaml](configs/registerjob.yaml)
```
aws ecr create-repository --repository-name ${REPOSITORY_NAME} --region $AWS_REGION --profile ${AWS_PROFILE} 
#private by default
#The repository with name 'cdi-hub/test-container' already exists in the registry with id '975050006673'
```

### Build and Upload container to the ECR [:link:](https://docs.aws.amazon.com/AmazonECR/latest/userguide/docker-push-ecr-image.html)
```
#cd Docker image location
IMAGE_ID=`aws ecr describe-repositories --repository-names ${REPOSITORY_NAME} --output text --query 'repositories[0].[repositoryUri]' --region $AWS_REGION --profile ${AWS_PROFILE}` 
docker build -t ${IMAGE_ID}:latest . 
aws ecr get-login-password --region ${AWS_REGION} --profile ${AWS_PROFILE} | docker login --username AWS --password-stdin "${IMAGE_ID}"  
docker push ${IMAGE_ID}:latest 
```

## Triggering job
* To list the images in a repository [:link:](https://docs.aws.amazon.com/cli/latest/reference/ecr/create-repository.html)
```
aws ecr list-images --repository-name ${REPOSITORY_NAME} --region ${AWS_REGION} --profile ${AWS_PROFILE}
```


## Blurs
### Launch and test image
```
docker run --name awsbatch_container --detach awsbatch:latest
docker exec -it awsbatch_container bash
```


