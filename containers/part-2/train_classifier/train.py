# train.py
import argparse
import os
import boto3
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-data-path', type=str, required=True)
    parser.add_argument('--model-save-path', type=str, required=True)
    
    args = parser.parse_args()

    # Load training data from S3
    s3 = boto3.client('s3')
    obj = s3.get_object(Bucket=args.train_data_path.split('/')[2], Key='/'.join(args.train_data_path.split('/')[3:]))
    df = pd.read_csv(obj['Body'])

    # Assume the last column is the target
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Create and train the pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression())
    ])
    pipeline.fit(X, y)

    # Save the trained model to S3
    with open('model.pkl', 'wb') as model_file:
        joblib.dump(pipeline, model_file)

    s3.upload_file('model.pkl', args.model_save_path.split('/')[2], '/'.join(args.model_save_path.split('/')[3:]))

if __name__ == '__main__':
    main()
