import boto3
import subprocess

def build_and_push_docker_image(repository_name):
    # Get the account ID and region for constructing the ECR image URI
    account_id = boto3.client('sts').get_caller_identity().get('Account')
    region = boto3.session.Session().region_name
    ecr_repository = f'{account_id}.dkr.ecr.{region}.amazonaws.com/{repository_name}'
    image_tag = ':latest'

    # Login to ECR
    login_cmd_output = subprocess.check_output(['aws', 'ecr', 'get-login-password']).strip().decode("utf-8")
    subprocess.check_call(f'docker login -u AWS -p {login_cmd_output} {account_id}.dkr.ecr.{region}.amazonaws.com', shell=True)

    # Build Docker image
    subprocess.check_call(f'docker build -t {repository_name} .', shell=True)

    # Tag the Docker image
    subprocess.check_call(f'docker tag {repository_name} {ecr_repository}{image_tag}', shell=True)

    # Create ECR repository if it doesn't exist
    ecr = boto3.client('ecr')
    try:
        ecr.describe_repositories(repositoryNames=[repository_name])
    except ecr.exceptions.RepositoryNotFoundException:
        ecr.create_repository(repositoryName=repository_name)

    # Push image to ECR
    subprocess.check_call(f'docker push {ecr_repository}{image_tag}', shell=True)

    print(f"Image pushed to {ecr_repository}{image_tag}")

repository_name = "autocoder_image"
build_and_push_docker_image(repository_name)
