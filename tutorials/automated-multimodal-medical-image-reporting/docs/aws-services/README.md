# AWS Services

## Introduction
A common approach to working with AWS involves first creating resources manually through the AWS Management Console for experimentation and validation. 
Once the setup is understood and tested, the process is automated using AWS CLI or Infrastructure as Code (IaC) tools such as CloudFormation, AWS CDK, or Terraform.

The true power of cloud computing lies in its "on-demand" nature, enabling you to easily create and delete resources as needed. 
This approach aligns with the philosophy of ["treating your servers like cattle, not pets"](https://devops.stackexchange.com/questions/653/what-is-the-definition-of-cattle-not-pets), focusing on scalability and disposability rather than individual care and maintenance.

## Setting up 

### Access to AWS access 
* Request AWS access to [Ben Thomas](https://github.com/bathomas)
* Open AWS access portal: https://ucl-cloud.awsapps.com/start#/

### aws-cli
* Installing or updating to the latest version of the AWS CLI under Ubuntu. For other OS see [here](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html)
```
sudo snap install aws-cli --classic
```

### Configure aws session following [this reference](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-sso.html)
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

### Setup workflow
1. [aws-batch](aws-bath)
2. [aws-s3-bucket](aws-s3-bucket)


### Cost 
* Elastic Container Registry
  * Pricing Storage settings https://calculator.aws/#/
15 GB per month x 0.10 USD = 1.50 USD. Elastic Container Registry pricing (monthly): 1.50 USD


## References
* Launch an Ubuntu EC2 instance using the AWS CLI: https://documentation.ubuntu.com/aws/en/latest/aws-how-to/instances/launch-ubuntu-ec2-instance/
* Create a Spot Instance request: https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/spot-requests.html
* Kumar, Nagresh, and Sanjay Kumar Sharma. "A Cost-Effective and Scalable Processing of Heavy Workload with AWS Batch." International Journal of Electrical and Electronics Research 10, no. 2 (2022): 144-149. [[PDF]](https://ijeer.forexjournal.co.in/papers-pdf/ijeer-100216.pdf) [[google-citations]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=A+Cost-Effective+and+Scalable+Processing+of+Heavy+Workload+with+AWS+Batch++&btnG=)


