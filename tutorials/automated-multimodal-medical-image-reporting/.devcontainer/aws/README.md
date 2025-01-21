# aws docker entry point

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
bash ../stop_container_and_removeit.bash
```

## aws
```
AWS_PROFILE=AWSAdministratorAccess-cdi-dev
bash aws-login.bash ${AWS_PROFILE}
#aws sso logout 
```

### Created Elastic Compute Cloud (Amazon EC2) orchestration 
https://eu-west-2.console.aws.amazon.com/batch/home?region=eu-west-2#wizard

```
* Configure job and orchestration type
  #Enable using Spot instances[can be interrupted with a two minute notification when EC2], 
  #VPC [aws-controltower-vpc]; 
* Create a compute environment
* Create a job queue
  #setup security group [BatchEnvironmentDefaultSG]; 
* Create a job definition
  #Create a job definition [Container configuration; No commands]) > amir-training-ec2-compute-env
* Create a job
```

### Build and Upload container to the ECR [:link:](https://docs.aws.amazon.com/AmazonECR/latest/userguide/docker-push-ecr-image.html)
```
#cd Docker image location
REPOSITORY_NAME="cdi-hub/aws-samples" 
AWS_REGION=eu-west-2 
IMAGE_ID=`aws ecr describe-repositories --repository-names ${REPOSITORY_NAME} --output text --query 'repositories[0].[repositoryUri]' --region $AWS_REGION --profile ${AWS_PROFILE}` 
docker build -t ${IMAGE_ID}:latest . 
aws ecr get-login-password --region ${AWS_REGION} --profile ${AWS_PROFILE} | docker login --username AWS --password-stdin "${IMAGE_ID}"  
docker push ${IMAGE_ID}:latest 
```




## Blurs
### Launch and test image
```
docker run --name awsbatch_container --detach awsbatch:latest
docker exec -it awsbatch_container bash
```


