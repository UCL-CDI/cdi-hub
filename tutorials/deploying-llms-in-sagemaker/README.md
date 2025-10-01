# Deploying and fine-tuning models on AWS

> A running document to showcase how to deploy and fine-tune DeepSeek R1 models with Hugging Face on AWS.
[ref](https://huggingface.co/blog/deepseek-r1-aws); [ref](https://aws.amazon.com/blogs/machine-learning/deploy-deepseek-r1-distilled-models-on-amazon-sagemaker-using-a-large-model-inference-container/)


## Editing notebook locally
* Install jupyter with uv
```
uv run --with jupyter jupyter lab
```
* Edit notebook in the browser


## Open AWS access portal

1. Go to https://ucl-cloud.awsapps.com/start#/
2. Select account and click AWSAdministratorAccess


## Managing Amazon SageMaker AI
1. Amazon SageMaker AI, creating QuickSetupDomain-${DATE_AND_TIME}
> Perfect for single user domains and first time users looking to get started with SageMaker.
Let Amazon SageMaker configure your account, and set up permissions for your SageMaker Domain.
* New IAM role with AmazonSageMakerFullAccess policy
* Public internet access, and standard encryption
* SageMaker Studio - New, and SageMaker Studio Classic integrations
* Sharable SageMaker Studio Notebooks
* SageMaker Canvas
* IAM Authentication
2. User profiles. Launch Studio
3. Jupiter Lab
    * Seeting up with Instance Type `ml.g6.2xlarge` with 1 of GPUs per replica for `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B	`
    * `ml.g5.12xlarge` from https://aws.amazon.com/blogs/machine-learning/deploy-deepseek-r1-distilled-models-on-amazon-sagemaker-using-a-large-model-inference-container/



## Create notebook instance

![fig](create-notebook-instance.png)


![fig](jupyter-lab.png)

## Terminate and delete services using console 
1. Space: cdi-hub-issue28. Delete app.
2. User Details: default-20250605T020811. Delete User
3. Domain: QuickSetupDomain-20250605T020811. Delete domain

## Delete an Amazon SageMaker AI domain (AWS CLI)
* Connect (click approve) to login
```bash
bash ../aws-services/scripts/aws-config-login.bash
```
* cleanup-sagemaker.bash
```bash
bash ../aws-services/scripts/cleanup-sagemaker.bash
```


## Clean and delete resources

* Open the Amazon S3 console at https://console.aws.amazon.com/s3/, and then delete the bucket that you created for storing model artifacts and the training dataset.
* Open the Amazon CloudWatch console at https://console.aws.amazon.com/cloudwatch/, and then delete all of the log groups that have names starting with /aws/sagemaker/.


## License
This [code repository] and the model weights](https://eu-west-2.console.aws.amazon.com/bedrock/home?region=eu-west-2#/model-catalog/bedrock-marketplace/deepseek-llm-r1-distill-qwen-1-5b) are licensed under the MIT License. DeepSeek-R1 series support commercial use, allow for any modifications and derivative works, including, but not limited to, distillation for training other LLMs. Please note that:

DeepSeek-R1-Distill-Qwen-1.5B, DeepSeek-R1-Distill-Qwen-7B, DeepSeek-R1-Distill-Qwen-14B and DeepSeek-R1-Distill-Qwen-32B are derived from Qwen-2.5 series , which are originally licensed under Apache 2.0 License , and now finetuned with 800k samples curated with DeepSeek-R1.
DeepSeek-R1-Distill-Llama-8B is derived from Llama3.1-8B-Base and is originally licensed under llama3.1 license .
DeepSeek-R1-Distill-Llama-70B is derived from Llama3.3-70B-Instruct and is originally licensed under llama3.3 license .


## Known Issues
* Quota
```
ResourceLimitExceeded: An error occurred (ResourceLimitExceeded) when calling the CreateEndpoint operation: The 
account-level service limit 'ml.g6.48xlarge for endpoint usage' is 0 Instances, with current utilization of 0 
Instances and a request delta of 1 Instances. Please use AWS Service Quotas to request an increase for this quota. 
If AWS Service Quotas is not available, contact AWS support to request an increase for this quota.
```


## References
* https://huggingface.co/blog/deepseek-r1-aws
* https://github.com/aws/sagemaker-huggingface-inference-toolkit
* https://github.com/aws/sagemaker-python-sdk
* https://repost.aws/knowledge-center/sagemaker-resource-utilization