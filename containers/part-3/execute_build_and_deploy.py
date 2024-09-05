import boto3
import sagemaker
from sagemaker.estimator import Estimator
from sagemaker.model import Model
from datetime import datetime
###########
def get_execution_role_sagemaker_studio():
    """Get the execution role for the current SageMaker Studio notebook.

    Returns:
        str: The ARN of the execution role.
    """
    session = sagemaker.Session()
    return session.get_caller_identity_arn()

role = get_execution_role_sagemaker_studio() # e.g., 'arn:aws:iam::account_id:role/service-role/role_name'
training_data_s3_uri = "<YOUR_TRAINING_DATA_S3_URI>" # e.g., 's3://my-bucket/path/to/training/data'

hyperparameters = {
    'model_name': model_name,
    'embedding_model_path': 's3://your-bucket/path-to-your-embedding-model/',
    'tokenizer_path': 's3://your-bucket/path-to-your-tokenizer/',
    'full_classifier_path': 's3://your-bucket/path-to-your-full-classifier/',
    'title_classifier_path': 's3://your-bucket/path-to-your-title-classifier/'
}


###########
current_date = datetime.now().strftime("%Y-%m-%d")
model_name = f"SSOCAutoCoder_{current_date}"
endpoint_name = f"SSOCAutoCoder_Endpoint_{current_date}"

###########

def train_and_deploy(repository_name, role, training_data_s3_uri, hyperparameters):
    # Construct the image URI
    account_id = boto3.client('sts').get_caller_identity().get('Account')
    region = boto3.session.Session().region_name
    ecr_repository = f'{account_id}.dkr.ecr.{region}.amazonaws.com/{repository_name}'
    image_uri = f"{ecr_repository}:latest"

    # Define SageMaker Estimator for training
    estimator = Estimator(
        image_uri=image_uri,
        role=role,
        instance_count=1,
        instance_type='ml.m4.xlarge',
        hyperparameters=hyperparameters
    )

    # Train
    estimator.fit({'training': training_data_s3_uri})

    # Construct the model and endpoint names
    current_date = datetime.now().strftime("%Y-%m-%d")

    # Deploy the model to create a prediction endpoint and print its name
    predictor = estimator.deploy(
        instance_type='ml.m4.xlarge',
        initial_instance_count=1,
        model_name=model_name,
        endpoint_name=endpoint_name
    )

    print(f"Endpoint name: {predictor.endpoint_name}")

train_and_deploy("repository_name", role, training_data_s3_uri, hyperparameters)