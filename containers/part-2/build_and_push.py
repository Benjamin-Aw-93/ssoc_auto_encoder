# build_and_push.py
import boto3
import subprocess

IMAGE_NAME = 'sklearn-training'
REPO_NAME = 'sklearn-training'

def build_and_push_docker_image():
    # Build the Docker image
    subprocess.run(['docker', 'build', '-t', IMAGE_NAME, '.'])
    
    # Authenticate Docker to ECR
    ecr_client = boto3.client('ecr')
    auth = ecr_client.get_authorization_token()
    token = auth['authorizationData'][0]['authorizationToken']
    username, password = token.split(':')
    subprocess.run(['docker', 'login', '-u', username, '-p', password, auth['authorizationData'][0]['proxyEndpoint']])
    
    # Tag and push the Docker image to ECR
    image_uri = f"{auth['authorizationData'][0]['proxyEndpoint']}/{REPO_NAME}:latest"
    subprocess.run(['docker', 'tag', IMAGE_NAME, image_uri])
    subprocess.run(['docker', 'push', image_uri])

    print(f"Image pushed to ECR: {image_uri}")
    return image_uri

if __name__ == "__main__":
    build_and_push_docker_image()
