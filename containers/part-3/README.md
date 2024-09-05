# Part-3: Build and Deploy

## Overview:
In this phase, the workflow combines the embedding model and classifier to build a unified model named `SSOCAutoCoder`. This model is then containerized and deployed as a SageMaker endpoint.

## Files & Directories:

```
.
├── DockerFile                      # Dockerfile for building the custom container for SageMaker
├── train_and_deploy.py             # Script to train and deploy the model on SageMaker
├── build_and_push.py               # Script to build the Docker container and push to Amazon ECR
└── autocoder
    ├── nginx.conf                  # Configuration for Nginx
    ├── train                       # Training script
    ├── serve                       # Script to start the model server
    ├── predictor.py                # Flask server for inference
    ├── wsgi.py                     # Gunicorn app entry point
    └── ssoc_autocoder
        └── combined_model.py       # Core model definition and methods

```

## Usage:

1. Ensure all dependencies specified in the `DockerFile` are available.
2. Update placeholders in the scripts with appropriate values.
3. Run `build_and_push.py` to build the Docker image and push it to AWS ECR.
4. Execute `execute_build_and_deploy.py` to build the combined model and deploy it as a SageMaker endpoint.

## Tips:
- Ensure that the SageMaker execution role has necessary permissions, such as accessing specific S3 paths, the ECR repository, and creating/managing SageMaker training jobs and endpoints.

## Note:
Make sure that the embeddings and classifier from the previous parts are available and have been trained before deploying in this part.