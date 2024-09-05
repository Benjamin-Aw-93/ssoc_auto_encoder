import boto3

def run_get_embedding(input_data_path, output_csv_path, model_output_path):
    client = boto3.client('sagemaker')

    response = client.create_transform_job(
        TransformJobName='your-transform-job-name',
        ModelName='your-model-name',  # This should be the name of your already created SageMaker model
        TransformInput={
            'DataSource': {
                'S3DataSource': {
                    'S3DataType': 'S3Prefix',
                    'S3Uri': input_data_path
                }
            },
            'ContentType': 'text/csv',
            'CompressionType': 'None',
            'SplitType': 'Line'
        },
        TransformOutput={
            'S3OutputPath': output_csv_path,
            'AssembleWith': 'Line'
        },
        TransformResources={
            'InstanceType': 'ml.m5.large',
            'InstanceCount': 1
        },
        Environment={
            'S3_MODEL_OUTPUT_PATH': model_output_path
        }
    )

    print(response)

if __name__ == '__main__':
    input_data_path = 's3://your-bucket/input-path/'
    output_csv_path = 's3://your-bucket/output-csv-path/'
    model_output_path = 's3://your-bucket/model-output-path/'

    run_get_embedding(input_data_path, output_csv_path, model_output_path)