# run_training_job.py
import boto3
import time

BUCKET_NAME = 'your-s3-bucket-name'
TRAIN_DATA_PATH = f's3://{BUCKET_NAME}/path/to/train/data.csv'
MODEL_SAVE_PATH = f's3://{BUCKET_NAME}/path/to/save/model/'
IMAGE_URI = 'YOUR_IMAGE_URI_FROM_ECR'  # Replace with the URI you obtained after pushing the image to ECR.

def run_sagemaker_training_job(image_uri):
    sm_client = boto3.client('sagemaker')
    job_name = f'sklearn-training-job-{int(time.time())}'
    
    create_response = sm_client.create_training_job(
        TrainingJobName=job_name,
        AlgorithmSpecification={
            'TrainingImage': image_uri,
            'TrainingInputMode': 'File'
        },
        RoleArn='YOUR_SAGEMAKER_ROLE_ARN',
        InputDataConfig=[
            {
                'ChannelName': 'train',
                'DataSource': {
                    'S3DataSource': {
                        'S3DataType': 'S3Prefix',
                        'S3Uri': TRAIN_DATA_PATH,
                        'S3DataDistributionType': 'FullyReplicated'
                    }
                }
            }
        ],
        OutputDataConfig={
            'S3OutputPath': MODEL_SAVE_PATH
        },
        ResourceConfig={
            'InstanceType': 'ml.m4.xlarge',
            'InstanceCount': 1,
            'VolumeSizeInGB': 10
        },
        StoppingCondition={
            'MaxRuntimeInSeconds': 3600
        }
    )
    print(f"Started SageMaker training job: {job_name}")
    return job_name

if __name__ == "__main__":
    run_sagemaker_training_job(IMAGE_URI)
