# Setting up AWS

## Intro
A common approach to working with AWS involves first creating resources manually through the AWS Management Console for experimentation and validation. 
Once the setup is understood and tested, the process is automated using AWS CLI or Infrastructure as Code (IaC) tools such as CloudFormation, AWS CDK, or Terraform.
The true power of cloud computing lies in its "on-demand" nature, enabling you to easily create and delete resources as needed. 
This approach aligns with the philosophy of ["treating your servers like cattle, not pets"](https://devops.stackexchange.com/questions/653/what-is-the-definition-of-cattle-not-pets), focusing on scalability and disposability rather than individual care and maintenance.

## Access to UCL cloud
* Request AWS access to [Ben Thomas](https://github.com/bathomas)
* Open AWS access portal: https://ucl-cloud.awsapps.com/start#/

## aws-cli
* Installing or updating to the latest version of the AWS CLI under Ubuntu. For other OS see [here](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html)
```
sudo snap install aws-cli --classic
```

## Configure aws session following [this reference](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-sso.html)
* Configuring IAM Identity Center authentication with the AWS CLI.
```
aws configure sso
#SSO session name (Recommended): AWSAdministratorAccess-cdi-dev
#SSO start URL [None]: https://ucl-cloud.awsapps.com/start
#SSO region [None]: eu-west-2
#SSO registration scopes [None]: sso:account:access
#Creating > .aws/config 
```
* Login/logout
```
AWS_PROFILE=AWSAdministratorAccess-cdi-dev
bash aws-login.bash ${AWS_PROFILE}
#aws sso logout 
```

## Batch workflow

See further details [here](../../.devcontainer/aws)

### Elastic Compute Cloud (Amazon EC2) orchestration [:link:](https://docs.aws.amazon.com/batch/latest/userguide/getting-started-ec2.html)

### Created compute environment 
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

### Elastic Container Registry
* Pricing Storage settings https://calculator.aws/#/
15 GB per month x 0.10 USD = 1.50 USD. Elastic Container Registry pricing (monthly): 1.50 USD
* Elastic Container Service


## References
* Launch an Ubuntu EC2 instance using the AWS CLI: https://documentation.ubuntu.com/aws/en/latest/aws-how-to/instances/launch-ubuntu-ec2-instance/
* Create a Spot Instance request: https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/spot-requests.html
* Kumar, Nagresh, and Sanjay Kumar Sharma. "A Cost-Effective and Scalable Processing of Heavy Workload with AWS Batch." International Journal of Electrical and Electronics Research 10, no. 2 (2022): 144-149. [[PDF]](https://ijeer.forexjournal.co.in/papers-pdf/ijeer-100216.pdf) [[google-citations]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=A+Cost-Effective+and+Scalable+Processing+of+Heavy+Workload+with+AWS+Batch++&btnG=)


