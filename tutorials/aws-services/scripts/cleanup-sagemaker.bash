#!/bin/bash
set -Eeu pipefail

# Set environment variables
export AWS_PROFILE="AWSAdministratorAccess-cdi-dev"
export REPOSITORY_NAME="cdi-hub/awsbatch-demo"
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export AWS_REGION="eu-west-2"


# Capture DomainId from list-domains
DOMAIN_ID=$(aws --region ${AWS_REGION} sagemaker list-domains \
  --query "Domains[0].DomainId" \
  --output text)

# Check if DOMAIN_ID is empty or None
if [ -z "$DOMAIN_ID" ] || [ "$DOMAIN_ID" == "None" ]; then
    echo "No domains found in region ${AWS_REGION}. Skipping further commands."
    exit 0
fi

# Retrieve the list of applications for the domain
aws --region ${AWS_REGION} sagemaker list-apps \
    --domain-id-equals ${DOMAIN_ID}

# Retrieve the list of user profiles in the domain
aws --region ${AWS_REGION} sagemaker list-user-profiles \
    --domain-id-equals ${DOMAIN_ID}




#Delete each shared space in the list.
# Get all SpaceNames for the domain
SPACE_NAMES=$(aws --region ${AWS_REGION} sagemaker list-spaces \
  --domain-id $DOMAIN_ID \
  --query "Spaces[].SpaceName" \
  --output text)

if [ -z "$SPACE_NAMES" ]; then
  echo "No spaces found for domain: $DOMAIN_ID"
else
  for SPACE_NAME in $SPACE_NAMES; do
    echo "Deleting space: $SPACE_NAME"
    aws --region ${AWS_REGION} sagemaker delete-space \
      --domain-id $DOMAIN_ID \
      --space-name $SPACE_NAME
  done
fi


# List all user profiles for the domain
USER_PROFILES=$(aws --region ${AWS_REGION} sagemaker list-user-profiles \
  --domain-id-equals $DOMAIN_ID \
  --query "UserProfiles[].UserProfileName" \
  --output text)

# Check if USER_PROFILES is empty
if [ -z "$USER_PROFILES" ]; then
  echo "No user profiles found for domain: $DOMAIN_ID"
else
  for USER_PROFILE_NAME in $USER_PROFILES; do
    echo "Deleting user profile: $USER_PROFILE_NAME"
    aws --region ${AWS_REGION} sagemaker delete-user-profile \
      --domain-id $DOMAIN_ID \
      --user-profile-name $USER_PROFILE_NAME
  done
fi



# Get current domain status
DOMAIN_STATUS=$(aws --region ${AWS_REGION} sagemaker describe-domain \
  --domain-id $DOMAIN_ID \
  --query "Status" \
  --output text)

# Only delete if not already Deleting or Deleted
if [ "$DOMAIN_STATUS" == "Deleting" ]; then
  echo "Domain $DOMAIN_ID is already being deleted. Skipping..."
elif [ "$DOMAIN_STATUS" == "Deleted" ]; then
  echo "Domain $DOMAIN_ID is already deleted. Skipping..."
else
  echo "Deleting domain $DOMAIN_ID..."
  aws --region ${AWS_REGION} sagemaker delete-domain \
    --domain-id $DOMAIN_ID \
    --retention-policy HomeEfsFileSystem=Delete
fi


