# Part-1: Generate Embeddings

## Overview:
This part of the workflow is responsible for generating embeddings using the `bert-base-uncased` model from Hugging Face's Transformers library.

## Files & Directories:

- `execute_get_embedding.py`: Script to initiate a SageMaker transform job to generate embeddings.
- `build_and_push.py`: Script to build a Docker image and push it to an AWS repository.
- `get_embedding/`: Directory containing resources for generating embeddings.
  - `requirements.txt`: List of Python packages required for the project.
  - `DockerFile`: Used to build the Docker container image.
  - `save_model.py`: Script to save the embedding model.
  - `get_embedding.py`: Script containing logic to generate embeddings.

## Usage:

1. Set appropriate values for placeholders in the scripts.
2. Run `build_and_push.py` to create the Docker image and push it to AWS.
3. Execute `execute_get_embedding.py` to initiate the SageMaker transform job.
