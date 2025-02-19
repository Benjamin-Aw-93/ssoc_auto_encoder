# General libraries
import pandas as pd
from copy import deepcopy
import numpy as np
from sklearn.model_selection import train_test_split
import time
from datetime import datetime

# Libraries for training on AWS
import os
import argparse

os.system('pip install transformers')

# Neural network libraries
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertModel, DistilBertTokenizer, DistilBertForSequenceClassification

def generate_encoding(reference_data, ssoc_colname = 'SSOC 2020'):

    """
    Generates encoding from SSOC 2020 to indices and vice versa.

    Creates a dictionary that maps 5D SSOCs (v2020) to indices
    as required by PyTorch for multi-class classification, as well
    as the opposite mapping for indices to SSOCs. This is to enable
    data to be used for training as well as for generation of new
    predictions based on unseen data.

    Args:
        reference_data: Pandas dataframe containing all SSOCs (v2020)
        ssoc_colname: Name of the SSOC column

    Returns:
        Dictionary containing the SSOC to index mapping (for preparing the
        dataset) and index to SSOC mapping (for interpreting the predictions),
        for each SSOC level from 1D to 5D.
    """

    # Initialise the dictionary object to store the encodings for each level
    encoding = {}

    # Iterate through each level from 1 to 5
    for level in range(1, 6):

        # Initialise a dictionary object to store the respective-way encodings
        ssoc_idx_mapping = {}

        # Slice the SSOC column by the level required, drop duplicates, and sort
        ssocs = list(np.sort(reference_data[ssoc_colname].astype('str').str.slice(0, level).unique()))

        # Iterate through each unique SSOC (at i-digit level) and add to dict
        for i, ssoc in enumerate(ssocs):
            ssoc_idx_mapping[ssoc] = i

        # Add each level's encodings to the output dictionary
        encoding[f'SSOC_{level}D'] = {

            # Store the SSOC to index encoding
            'ssoc_idx': ssoc_idx_mapping,
            # Store the index to SSOC encoding
            'idx_ssoc': {v: k for k, v in ssoc_idx_mapping.items()}
        }

    return encoding

def encode_dataset(data,
                   encoding,
                   colnames):
    """
    Encodes SSOCs into indices for the training data.

    Uses the generated encoding to encode the SSOCs into indices
    at each digit level for PyTorch training. Drops any other
    columns that are not relevant to the model training process.

    Args:
        data: Pandas dataframe of the training data with the correct SSOC
        encoding: Encoding for each SSOC level
        colnames: Dictionary of all column names

    Returns:
        Pandas dataframe with each digit SSOC encoded correctly
    """

    # Create a copy of the dataframe and filter out any nulls
    encoded_data = deepcopy(data)[data[colnames['SSOC']].notnull()]

    # Subset the dataframe for only the required 2 columns
    encoded_data = encoded_data[[colnames['job_description'], colnames['SSOC']]]
    encoded_data.columns = ['Text', 'SSOC']

    # For each digit, encode the SSOC correctly
    for ssoc_level, encodings in encoding.items():
        encoded_data[ssoc_level] = encoded_data['SSOC'].astype('str').str.slice(0, int(ssoc_level[5])).replace(encodings['ssoc_idx'])

    return encoded_data

# Create a new Python class to handle the additional complexity
class SSOC_Dataset(Dataset):
    """

    """

    # Define the class attributes
    def __init__(self, dataframe, tokenizer, max_len, colnames):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len

    # Define the iterable over the Dataset object
    def __getitem__(self, index):

        # Extract the text
        # NEED TO FIX THIS
        text = self.data.Text[index]

        # Pass in the data into the tokenizer
        inputs = self.tokenizer(
            text = text,
            text_pair = None,
            add_special_tokens = True,
            max_length = self.max_len,
            padding = 'max_length',
            return_token_type_ids = True,
            truncation = True
        )

        # Extract the IDs and attention mask
        ids = inputs['input_ids']
        mask = inputs['attention_mask']

        # Return all the outputs needed for training and evaluation
        return {
            'ids': torch.tensor(ids, dtype = torch.long),
            'mask': torch.tensor(mask, dtype = torch.long),
            'SSOC_1D': torch.tensor(self.data.SSOC_1D[index], dtype = torch.long),
            'SSOC_2D': torch.tensor(self.data.SSOC_2D[index], dtype = torch.long),
            'SSOC_3D': torch.tensor(self.data.SSOC_3D[index], dtype = torch.long),
            'SSOC_4D': torch.tensor(self.data.SSOC_4D[index], dtype = torch.long),
            'SSOC_5D': torch.tensor(self.data.SSOC_5D[index], dtype = torch.long),
        }

    # Define the length attribute
    def __len__(self):
        return self.len

def prepare_data(encoded_data,
                 tokenizer,
                 colnames,
                 parameters):
    """
    Prepares the encoded data for training and validation.



    Args:
        encoded_data:
        colnames: Dictionary of column names to
        parameters:

    Returns:

    """

    # Split the dataset into training and validation with a 20% split
    training_data, validation_data = train_test_split(encoded_data,
                                                      test_size = 0.2,
                                                      random_state = 2021)
    training_data.reset_index(drop = True, inplace = True)
    validation_data.reset_index(drop = True, inplace = True)

    # Creating the dataset and dataloader for the neural network
    training_loader = DataLoader(SSOC_Dataset(training_data, tokenizer, parameters['sequence_max_length'], colnames),
                                 batch_size = parameters['training_batch_size'],
                                 num_workers = parameters['num_workers'],
                                 shuffle = True,
                                 pin_memory = True)
    validation_loader = DataLoader(SSOC_Dataset(validation_data, tokenizer, parameters['sequence_max_length'], colnames),
                                   batch_size = parameters['training_batch_size'],
                                   num_workers = parameters['num_workers'],
                                   shuffle = True,
                                   pin_memory = True)

    return training_loader, validation_loader

class HierarchicalSSOCClassifier(torch.nn.Module):

    def __init__(self, training_parameters, encoding):

        self.training_parameters = training_parameters
        self.encoding = encoding

        # Initialise the class, not sure exactly what this does
        super(HierarchicalSSOCClassifier, self).__init__()

        # Load the DistilBert model which will generate the embeddings
        self.l1 = DistilBertModel.from_pretrained(self.training_parameters['pretrained_model'])

        for param in self.l1.parameters():
            param.requires_grad = False

        # Generate counts of each digit SSOCs
        SSOC_1D_count = len(self.encoding['SSOC_1D']['ssoc_idx'].keys())
        SSOC_2D_count = len(self.encoding['SSOC_2D']['ssoc_idx'].keys())
        SSOC_3D_count = len(self.encoding['SSOC_3D']['ssoc_idx'].keys())
        SSOC_4D_count = len(self.encoding['SSOC_4D']['ssoc_idx'].keys())
        SSOC_5D_count = len(self.encoding['SSOC_5D']['ssoc_idx'].keys())

        # Stack 1: Predicting 1D SSOC (9)
        if self.training_parameters['max_level'] >= 1:
            self.ssoc_1d_stack = torch.nn.Sequential(
                torch.nn.Linear(768, 768),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.3),
                torch.nn.Linear(768, 768),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.3),
                torch.nn.Linear(768, 128),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.3),
                torch.nn.Linear(128, SSOC_1D_count)
            )

        # Stack 2: Predicting 2D SSOC (42)
        if self.training_parameters['max_level'] >= 2:

            # Adding the predictions from Stack 1 to the word embeddings
            # n_dims_2d = 777
            n_dims_2d = 768 + SSOC_1D_count

            self.ssoc_2d_stack = torch.nn.Sequential(
                torch.nn.Linear(n_dims_2d, n_dims_2d),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.3),
                torch.nn.Linear(n_dims_2d, n_dims_2d),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.3),
                torch.nn.Linear(n_dims_2d, 256),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.3),
                torch.nn.Linear(256, 128),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.3),
                torch.nn.Linear(128, SSOC_2D_count)
            )

        # Stack 3: Predicting 3D SSOC (144)
        if self.training_parameters['max_level'] >= 3:

            # Adding the predictions from Stacks 1 and 2 to the word embeddings
            # n_dims_3d = 819
            n_dims_3d = 768 + SSOC_1D_count + SSOC_2D_count

            self.ssoc_3d_stack = torch.nn.Sequential(
                torch.nn.Linear(n_dims_3d, n_dims_3d),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.3),
                torch.nn.Linear(n_dims_3d, n_dims_3d),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.3),
                torch.nn.Linear(n_dims_3d, n_dims_3d),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.3),
                torch.nn.Linear(n_dims_3d, 512),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.3),
                torch.nn.Linear(512, 256),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.3),
                torch.nn.Linear(256, SSOC_3D_count)
            )

        # Stack 4: Predicting 4D SSOC (413)
        if self.training_parameters['max_level'] >= 4:

            # Adding the predictions from Stacks 1, 2, and 3 to the word embeddings
            # n_dims_4d = 963
            n_dims_4d = 768 + SSOC_1D_count + SSOC_2D_count + SSOC_3D_count

            self.ssoc_4d_stack = torch.nn.Sequential(
                torch.nn.Linear(n_dims_4d, n_dims_4d),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.3),
                torch.nn.Linear(n_dims_4d, n_dims_4d),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.3),
                torch.nn.Linear(n_dims_4d, n_dims_4d),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.3),
                torch.nn.Linear(n_dims_4d, 768),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.3),
                torch.nn.Linear(768, 512),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.3),
                torch.nn.Linear(512, SSOC_4D_count)
            )

        # Stack 5: Predicting 5D SSOC (997)
        if self.training_parameters['max_level'] >= 5:

            # Adding the predictions from Stacks 1, 2, and 3 to the word embeddings
            # n_dims_5d = 1376
            n_dims_5d = 768 + SSOC_1D_count + SSOC_2D_count + SSOC_3D_count + SSOC_4D_count

            self.ssoc_5d_stack = torch.nn.Sequential(
                torch.nn.Linear(n_dims_5d, n_dims_5d),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.3),
                torch.nn.Linear(n_dims_5d, n_dims_5d),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.3),
                torch.nn.Linear(n_dims_5d, n_dims_5d),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.3),
                torch.nn.Linear(n_dims_5d, n_dims_5d),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.3),
                torch.nn.Linear(n_dims_5d, SSOC_5D_count)
            )


    def forward(self, input_ids, attention_mask):

        # Obtain the sentence embeddings from the DistilBERT model
        embeddings = self.l1(input_ids = input_ids, attention_mask = attention_mask)
        hidden_state = embeddings[0]
        X = hidden_state[:, 0]

        # Initialise a dictionary to hold all the predictions
        predictions = {}

        # 1D Prediction
        if self.training_parameters['max_level'] >= 1:
            predictions['SSOC_1D'] = self.ssoc_1d_stack(X)

        # 2D Prediction
        if self.training_parameters['max_level'] >= 2:
            X = torch.cat((X, predictions['SSOC_1D']), dim = 1)
            predictions['SSOC_2D'] = self.ssoc_2d_stack(X)

        # 3D Prediction
        if self.training_parameters['max_level'] >= 3:
            X = torch.cat((X, predictions['SSOC_2D']), dim = 1)
            predictions['SSOC_3D'] = self.ssoc_3d_stack(X)

        # 4D Prediction
        if self.training_parameters['max_level'] >= 4:
            X = torch.cat((X, predictions['SSOC_3D']), dim = 1)
            predictions['SSOC_4D'] = self.ssoc_4d_stack(X)

        # 5D Prediction
        if self.training_parameters['max_level'] >= 5:
            X = torch.cat((X, predictions['SSOC_4D']), dim = 1)
            predictions['SSOC_5D'] = self.ssoc_5d_stack(X)

        return {f'SSOC_{i}D': predictions[f'SSOC_{i}D'] for i in range(1, self.training_parameters['max_level'] + 1)}

def prepare_model(encoding, parameters):
    """

    Args:
        encoding:
        parameters:

    Returns:

    """

    model = HierarchicalSSOCClassifier(parameters, encoding)
    model.to(parameters['device'])
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(params = model.parameters(), lr = parameters['learning_rate'])
    optimizer.zero_grad()

    return model, loss_function, optimizer

def calculate_accuracy(big_idx, targets):
    n_correct = (big_idx == targets).sum().item()
    return n_correct

def train_model(model,
                loss_function,
                optimizer,
                training_loader,
                validation_loader,
                parameters):

    # Start the timer
    start_time = datetime.now()
    print(f"Training started on: {start_time.strftime('%d %b %Y - %H:%M:%S')}")
    print("====================================================================")

    # Iterate over each training epoch
    for epoch_num in range(1, parameters['epochs']+1):

        # Start the timer for the epoch
        epoch_start_time = datetime.now()
        print(f"> Epoch {epoch_num} started on: {epoch_start_time.strftime('%d %b %Y - %H:%M:%S')}")
        print("--------------------------------------------------------------------")

        tr_loss = 0
        tr_n_correct = 0
        nb_tr_steps = 0
        nb_tr_examples = 0

        # Set the NN to train mode
        model.train()

        # Start the timer for the batch
        batch_start_time = datetime.now()

        # Iterate over each batch
        for batch, data in enumerate(training_loader):

            # Extract the data
            ids = data['ids'].to(parameters['device'], dtype = torch.long)
            mask = data['mask'].to(parameters['device'], dtype = torch.long)

            # Run the forward prop
            predictions = model(ids, mask)

            # Iterate through each SSOC level
            for ssoc_level, preds in predictions.items():

                # Extract the correct target for the SSOC level
                targets = data[ssoc_level].to(parameters['device'], dtype = torch.long)

                # Compute the loss function using the predictions and the targets
                level_loss = loss_function(preds, targets)

                # Initialise the loss variable if this is the 1D level
                # Else add to the loss variable
                # Note the weights on each level
                if ssoc_level == 'SSOC_1D':
                    loss = level_loss * parameters['loss_weights'][ssoc_level]
                else:
                    loss += level_loss * parameters['loss_weights'][ssoc_level]

            # Use the deepest level predictions to calculate accuracy
            # Exploit the fact that the last preds object is the deepest level one
            top_probs, top_probs_idx = torch.max(preds.data, dim = 1)
            tr_n_correct += calculate_accuracy(top_probs_idx, targets)

            # Add this batch's loss to the overall training loss
            tr_loss += loss.item()

            # Keep count for the batch steps and number of examples
            nb_tr_steps += 1
            nb_tr_examples += targets.size(0)

            # Run backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # For every X number of steps, print some information
            batch_size_printing = 200
            if (batch + 1) % batch_size_printing == 0:
                loss_step = tr_loss / nb_tr_steps
                accu_step = (tr_n_correct * 100) / nb_tr_examples
                print(f">> Training Loss per {batch_size_printing} steps: {loss_step:.4f} ")
                print(f">> Training Accuracy per {batch_size_printing} steps: {accu_step:.2f}%")
                print(f">> Batch of {batch_size_printing} took {(datetime.now() - batch_start_time).total_seconds() / 60:.2f} mins")
                batch_start_time = datetime.now()

        # Set the model to evaluation mode for the validation
        model.eval()

        va_n_correct = 0
        va_loss = 0
        nb_va_steps = 0
        nb_va_examples = 0

        # Disable the calculation of gradients
        with torch.no_grad():

            # Iterate over each batch
            for batch, data in enumerate(validation_loader):

                # Extract the data
                ids = data['ids'].to(parameters['device'], dtype = torch.long)
                mask = data['mask'].to(parameters['device'], dtype = torch.long)

                # Run the forward prop
                predictions = model(ids, mask)

                # Iterate through each SSOC level
                for ssoc_level, preds in predictions.items():

                    # Extract the correct target for the SSOC level
                    targets = data[ssoc_level].to(parameters['device'], dtype = torch.long)

                    # Compute the loss function using the predictions and the targets
                    level_loss = loss_function(preds, targets)

                    # Initialise the loss variable if this is the 1D level
                    # Else add to the loss variable
                    # Note the weights on each level
                    if ssoc_level == 'SSOC_1D':
                        loss = level_loss * parameters['loss_weights'][ssoc_level]
                    else:
                        loss += level_loss * parameters['loss_weights'][ssoc_level]

                # Use the deepest level predictions to calculate accuracy
                # Exploit the fact that the last preds object is the deepest level one
                top_probs, top_probs_idx = torch.max(preds.data, dim = 1)
                va_n_correct += calculate_accuracy(top_probs_idx, targets)

                # Add this batch's loss to the overall training loss
                va_loss += loss.item()

                # Keep count for the batch steps and number of examples
                nb_va_steps += 1
                nb_va_examples += targets.size(0)

        print("--------------------------------------------------------------------")
        epoch_tr_loss = tr_loss / nb_tr_steps
        epoch_va_loss = va_loss / nb_va_steps
        epoch_tr_accu = (tr_n_correct * 100) / nb_tr_examples
        epoch_va_accu = (va_n_correct * 100) / nb_va_examples
        print(f"> Epoch {epoch_num} Loss = \tTraining: {epoch_tr_loss:.3f}  \tValidation: {epoch_va_loss:.3f}")
        print(f"> Epoch {epoch_num} Accuracy = \tTraining: {epoch_tr_accu:.2f}%  \tValidation: {epoch_va_accu:.2f}%")
        print(f"> Epoch {epoch_num} took {(datetime.now() - epoch_start_time).total_seconds() / 60:.2f} mins")
        print("====================================================================")

    end_time = datetime.now()
    print("Training ended on:", end_time.strftime("%d %b %Y - %H:%M:%S"))
    print(f"Total training time: {(end_time - start_time).total_seconds() / 3600:.2f} hours")

    return

def generate_prediction(model,
                        tokenizer,
                        text,
                        target,
                        parameters):

    tokenized = tokenizer(
        text = text,
        text_pair = None,
        add_special_tokens = True,
        max_length = parameters['sequence_max_length'],
        padding = 'max_length',
        return_token_type_ids = True,
        truncation = True
    )
    test_ids = torch.tensor([tokenized['input_ids']], dtype = torch.long)
    test_mask = torch.tensor([tokenized['attention_mask']], dtype = torch.long)

    model.eval()
    preds = model(test_ids, test_mask)
    m = torch.nn.Softmax(dim = 1)

    predicted_1D = encoding['SSOC_1D']['idx_ssoc'][np.argmax(preds["SSOC_1D"].detach().numpy())]
    predicted_1D_proba = np.max(m(preds['SSOC_1D']).detach().numpy())
    predicted_2D = encoding['SSOC_2D']['idx_ssoc'][np.argmax(preds["SSOC_2D"].detach().numpy())]
    predicted_2D_proba = np.max(m(preds['SSOC_2D']).detach().numpy())

    print(f"Target: {target}")
    print(f"Model predicted 1D: {predicted_1D} ({predicted_1D_proba * 100:.2f}%)")
    print(f"Model predicted 2D: {predicted_2D} ({predicted_2D_proba * 100:.2f}%)")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument('--workers', type = int, default = 2)
    parser.add_argument('--epochs', type = int, default = 10)
    parser.add_argument('--batch-size', type = int, default = 64)
    parser.add_argument('--learning-rate', type = float, default = 0.005)
    parser.add_argument('--use-cuda', type = bool, default = True)

    # Data, model, and output directories
    parser.add_argument('--output-data-dir', type = str, default = os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type = str, default = os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type = str, default = os.environ['SM_CHANNEL_TRAINING'])
    #parser.add_argument('--train', type = str, default = os.environ['SM_CHANNEL_TRAIN'])
    #parser.add_argument('--test', type = str, default = os.environ['SM_CHANNEL_TEST'])

    args, _ = parser.parse_known_args()

    colnames = {
        'SSOC': 'SSOC 2020',
        'job_description': 'Cleaned_Description'
    }

    parameters = {
        'sequence_max_length': 512,
        'max_level': 5,
        'training_batch_size': 32,
        'validation_batch_size': 32,
        'epochs': 10,
        'learning_rate': 0.0001,
        'pretrained_model': 'distilbert-base-uncased',
        'num_workers': 2,
        'loss_weights': {
            'SSOC_1D': 20,
            'SSOC_2D': 5,
            'SSOC_3D': 3,
            'SSOC_4D': 2,
            'SSOC_5D': 1
        },
        'device': 'cuda'
    }

    #train = pd.read_csv(os.path.join(args.data_dir, 'train-7oct.csv'))
    #test = pd.read_csv(os.path.join(args.data_dir, 'test-7oct.csv'))
    data = pd.read_csv(os.path.join(args.data_dir, 'train_full.csv'))
    SSOC_2020 = pd.read_csv(os.path.join(args.data_dir, 'SSOC_2020.csv'))

    encoding = generate_encoding(SSOC_2020)
    encoded_data = encode_dataset(data, encoding, colnames)
    tokenizer = DistilBertTokenizer.from_pretrained(parameters['pretrained_model'])
    training_loader, validation_loader = prepare_data(encoded_data, tokenizer, colnames, parameters)
    model, loss_function, optimizer = prepare_model(encoding, parameters)

    print("Loading pretrained model...")
    model.load_state_dict(torch.load(os.path.join(args.data_dir, 'sm-model-5D-1epoch.pth')))
    print("Model loaded successfully!")

    train_model(model, loss_function, optimizer, training_loader, validation_loader, parameters)

    model_path = os.path.join(args.model_dir, 'sm-model-5D-10epoch.pth')

    # Move the best model to cpu and resave it
    with open(model_path, 'wb') as f:
        torch.save(model.cpu().state_dict(), f)