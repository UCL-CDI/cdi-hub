# S3 Bucket 
* Log in
```
bash ../scripts/aws-login.bash 
```
* Log in into your AWS access portal
Select either `cdi-innov-dev` or `arc-playpen-collaborations` and launch AWSAdministratorAccess: 
https://ucl-cloud.awsapps.com/start

* Create a bucket 
```
BUCKET_NAME=amir-training
BUCKET_POSTFIX=$(uuidgen --random | cut -d'-' -f1)
export BUCKET_ROOT=${BUCKET_NAME}-${BUCKET_POSTFIX}
aws s3 mb s3://${BUCKET_ROOT} --profile ${AWS_PROFILE}
```
* Process data in S3 buckets: Upload Data to S3 Bucket URI
Use aws s3 sync command to copy files. Use --dryrun to check what files are going to be copied 
```
#cd datapath
aws s3 sync . s3://amir-training-b70c6730 --dryrun --profile ${AWS_PROFILE} 
```
* List files 
```
aws s3 ls s3://amir-training-b70c6730 --profile ${AWS_PROFILE} 
``` 
* Remove files
```
aws s3 rm s3://amir-training-b70c6730/1_IM-0001-3001.dcm.png --profile ${AWS_PROFILE}
```
