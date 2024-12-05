# Setting up AWS

## Intro
A common approach to working with AWS involves first creating resources manually through the AWS Management Console for experimentation and validation. 
Once the setup is understood and tested, the process is automated using AWS CLI or Infrastructure as Code (IaC) tools such as CloudFormation, AWS CDK, or Terraform.
The true power of cloud computing lies in its "on-demand" nature, enabling you to easily create and delete resources as needed. 
This approach aligns with the philosophy of ["treating your servers like cattle, not pets"](https://devops.stackexchange.com/questions/653/what-is-the-definition-of-cattle-not-pets), focusing on scalability and disposability rather than individual care and maintenance.

## Access to UCL cloud
1. Request AWS access to [Ben Thomas](https://github.com/bathomas)
2. Open AWS access portal: https://ucl-cloud.awsapps.com/start#/

## Accounts in the AWS access portal

### Create security group
1. you would need to go to your AWS access portal to select your account to and `AWSPowerUserAccess`.
2. Select zone (e.g. London)
3. Select [EC2](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/concepts.html)  
4. Select Network & Security to create a [security group](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-security-groups.html?icmpid=docs_ec2_console#creating-security-group) where you need to choose:
    * Security group name Info
    * Security group description 
    * [VPC](https://docs.aws.amazon.com/vpc/latest/userguide/what-is-amazon-vpc.html), 
    * Select either default or customised [inbound and outbound rules](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/security-group-rules-reference.html?icmpid=docs_ec2_console) 
    * `Create security group`.

### Launch an instance
Generally, the following steps should be followed. For more details, refer to the information provided [here](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-launch-instance-wizard.html?icmpid=docs_ec2_console). 
1. Name
2. Select application and OS Images (Amazon Machine Image) 
3. Instance type
4. Key pair. Existing key pair, Create a new pair and Proceed without key pair.
5. Network settings, selecting existing security group.


### Delete instance 
Select instance and terminate (delete instance)

### Connect to instance?

### Launch local instance
1. Installing or updating to the latest version of the AWS CLI under Ubuntu
```
sudo snap install aws-cli --classic
```
2. `aws configure`
* AWS Access Key ID [None]: :warning: "Where to find it?"
* AWS Secret Access Key [None]: :warning:  "Where to find it?"
* Default region name [None]: :warning:  "Where to find it?"
* Default output format [None]: :warning:  "Where to find it?"

3. To create a Spot Instance request using run-instances
```
aws ec2 run-instances \
    --image-id ami-0abcdef1234567890 \
    --instance-type t2.micro \
    --count 5 \
    --subnet-id subnet-08fc749671b2d077c \
    --key-name MyKeyPair \
    --security-group-ids sg-0b0384b66d7d692f9 \
    --instance-market-options file://spot-options.json
```

## References
* Launch an Ubuntu EC2 instance using the AWS CLI: https://documentation.ubuntu.com/aws/en/latest/aws-how-to/instances/launch-ubuntu-ec2-instance/
* Create a Spot Instance request: https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/spot-requests.html



