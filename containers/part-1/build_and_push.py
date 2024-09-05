import boto3
import subprocess
import os

def build_and_push():
    # Specify your repository name and image name
    REPOSITORY_NAME = "your-repo-name"
    IMAGE_NAME = "your-image-name"
    
    # Get AWS account ID
    client = boto3.client('sts')
    AWS_ACCOUNT_ID = client.get_caller_identity()['Account']

    os.chdir("get_embedding")
    
    # Build the docker image
    subprocess.run(["docker", "build", "-t", IMAGE_NAME, "."])

    # Tag the docker image
    docker_tag = f"{AWS_ACCOUNT_ID}.dkr.ecr.your-region.amazonaws.com/{REPOSITORY_NAME}:{IMAGE_NAME}"
    subprocess.run(["docker", "tag", IMAGE_NAME, docker_tag])

    # Login to ECR
    login_response = subprocess.check_output(
        ["aws", "ecr", "get-login", "--no-include-email", "--region", "your-region"],
        universal_newlines=True
    )
    subprocess.run(login_response.split())

    # Push the docker image to ECR
    subprocess.run(["docker", "push", docker_tag])

if __name__ == '__main__':
    build_and_push()
