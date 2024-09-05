# Part-2: Train Classifier

## Overview:
This part of the workflow is focused on training a classifier using the embeddings generated in `part-1`.

## Files & Directories:

- `execute_train_classifier.py`: Script to initiate a SageMaker training job for the classifier.
- `build_and_push.py`: Script to build a Docker image and push it to an AWS repository.
- `train_classifier/`: Directory containing resources for training the classifier.
  - `DockerFile`: Used to build the Docker container image for training.
  - `train.py`: Script containing the logic to train the classifier.

## Usage:
1. Ensure that the required dependencies are installed as per the `DockerFile`.
2. Set appropriate values for placeholders in the scripts.
3. Run `build_and_push.py` to create the Docker image and push it to AWS.
4. Execute `execute_train_classifier.py` to initiate the SageMaker training job.

## Note:
Ensure that the embeddings from `part-1` are available and accessible for the training process in this part.
