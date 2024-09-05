# Workflow for Generating Embeddings, Training Embedding Classifer, Building SSOCAutoCoder and Deploying SSOCAutoCoder

This repository contains a structured workflow that encompasses the generation of embeddings, training of a classifier, and the deployment of the combined model.

`SSOCAutoCoder` consists of the `EmbeddingModel` and `EmbeddingClassifer`

## Structure:

- `part-1`: Focuses on generating embeddings.
- `part-2`: Concentrates on training a classifier using the generated embeddings.
- `part-3`: Combines the embedding model and classifier, then builds and deploys the combined model.

## Part-1: Generate Embeddings

In this phase, embeddings are generated using the `bert-base-uncased` model from Hugging Face's Transformers library.

### Usage:

1. Replace placeholders in the scripts with the appropriate values.
2. Execute `build_and_push.py` to create the Docker image and push it to AWS.
3. Run `execute_get_embedding.py` to initiate the SageMaker transform job.

[Detailed README for Part-1](./part-1/README.md)

## Part-2: Train Classifier

This part trains a classifier using the embeddings produced in `part-1`.

### Usage:

1. Install the required dependencies as per the `DockerFile`.
2. Update placeholders in the scripts.
3. Execute `build_and_push.py` to build the Docker image and push it to AWS.
4. Run `execute_train_classifier.py` to start the SageMaker training job.

[Detailed README for Part-2](./part-2/README.md)

## Part-3: Build and Deploy

In this phase, the workflow integrates the embedding model and classifier to build a unified model named `SSOCAutoCoder`. This model is then containerized and deployed as a SageMaker endpoint.

### Usage:

1. Ensure all dependencies specified in the `DockerFile` are installed.
2. Modify the scripts by replacing placeholders with appropriate values.
3. Run `build_and_push.py` to build the Docker image and push it to AWS ECR.
4. Execute `execute_build_and_deploy.py` to build the combined model and deploy it as a SageMaker endpoint.

[Detailed README for Part-3](./part-3/README.md)

## Overall Tips:

- Ensure necessary permissions for the SageMaker execution role, such as accessing specific S3 paths, the ECR repository, and creating/managing SageMaker training jobs and endpoints.
- It's recommended to follow the workflow sequentially: Start with `part-1`, proceed to `part-2`, and conclude with `part-3`.
