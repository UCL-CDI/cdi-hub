#!/bin/bash
set -ex

mkdir input-data

S3_BUCKET_PATH=s3://amir-training-b70c6730/test-sample
aws s3 cp --recursive ${S3_BUCKET_PATH} /input-data

ls -l > test_output.txt
aws s3 cp test_output.txt s3://amir-training-b70c6730/test-output

#TO UPDATE for the case of bigger data size
#https://seqera.io/blog/mountpoint-for-amazon-s3-vs-fusion-file-system/
