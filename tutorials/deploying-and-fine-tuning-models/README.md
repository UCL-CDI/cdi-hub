# Deploying and fine-tuning models on AWS

## Managing Amazon SageMaker AI
1. Amazon SageMaker AI, creating QuickSetupDomain-20250605T020811
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
Seeting up with Instance Type `ml.g6.2xlarge` with 1 of GPUs per replica for `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B	`

![fig](jupyter-lab.png)

Terminate and delete services
1. Space: cdi-hub-issue28. Delete app.
2. User Details: default-20250605T020811. Delete User
3. Domain: QuickSetupDomain-20250605T020811. Delete domain



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
https://huggingface.co/blog/deepseek-r1-aws
https://github.com/aws/sagemaker-huggingface-inference-toolkit
https://github.com/aws/sagemaker-python-sdk