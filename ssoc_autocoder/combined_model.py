"""
SSOCAutoCoder Module

This module contains the `SSOCAutoCoder` class which facilitates automatic SSOC labelling using embeddings generated from text.
The class provides functionality to build, predict, and save the model.

Example Usage #1:
from ssoc_autocoder.combined_model import SSOCAutoCoder
SSOC = SSOCAutoCoder()
SSOC.build(
    model_name="test",
    embedding_model_path="Models/language_model",
    tokenizer_path="Models/distilbert-tokenizer-pretrained-7epoch",
    full_classifier_path="Models/230829_1231h_full_logreg.pkl",
    title_classifier_path="Models/230827_2252h_title_logreg.pkl"
)
SSOC.predict(titles=["software engineer", "data scientist", "machine learning engineer"], descriptions=["build fullstack applications", "builds dashboards", "builds AI models"], top_n=3)
SSOC.predict_lite(titles=["software engineer", "data scientist", "machine learning engineer"], top_n=3)


Example Usage #2:
from ssoc_autocoder.combined_model import SSOCAutoCoder
SSOC = SSOCAutoCoder()
SSOC.build(
    model_name="test",
    full_classifier_path="Models/bge-base-en_w_jd_log_reg_pipeline.pkl",
    title_classifier_path="Models/bge-base-en_log_reg_pipeline.pkl",
    hugging_face_model_name = "BAAI/bge-base-en",
    from_hugging_face = True
)
SSOC.predict(titles=["software engineer", "data scientist", "machine learning engineer"], descriptions=["build fullstack applications", "builds dashboards", "builds AI models"], top_n=3)
SSOC.predict_lite(titles=["software engineer", "data scientist", "machine learning engineer"], top_n=3)
SSOC.save("Models/autocoder.pkl")

"""

import numpy as np
import pandas as pd
import json
import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer
from sklearn.base import BaseEstimator
from typing import Union
import pickle
from datetime import datetime

class SSOCAutoCoder:
    """
    The SSOCAutoCoder class provides automatic coding based on embeddings generated from text.
    
    Attributes:
        - embedding_model: The model used for generating embeddings.
        - tokenizer: Tokenizer for processing text.
        - full_classifier: Classifier for generating predictions based on full text (title + description).
        - title_classifier: Classifier for generating predictions based on title only.
        - name: A name assigned to the model for identification purposes.
        - framework: Framework used, e.g., 'pytorch'.

    Methods:
        - build(): Constructs the autocoder model.
        - predict(): Uses the full_classifier to predict a code based on a given title and description.
        - predict_lite(): Uses the title_classifier to predict a code based on a given title.
        - save(): Saves the autocoder state to a file.
    """
    
    def __init__(self):
        # Initializations for the auto-coder
        self.embedding_model = None
        self.tokenizer = None
        self.full_classifier = None
        self.title_classifier = None
        self.name = None
        self.framework = None

    def _check_classifier_validity(self, classifier):
        """
        Check if the classifier has the necessary attributes and methods.
        """
        required_methods = ['predict', 'predict_proba']
        required_attributes = ['classes_', 'n_features_in_']

        for method in required_methods:
            if not callable(getattr(classifier, method, None)):
                raise ValueError(f"Classifier must have a callable '{method}' method.")

        for attribute in required_attributes:
            if not hasattr(classifier, attribute):
                raise ValueError(f"Classifier must have the '{attribute}' attribute.")

    def build(self,
              model_name: str,
              full_classifier_path: str,
              title_classifier_path: str,
              embedding_model_path: str = None,
              tokenizer_path: str = None,
              hugging_face_model_name: str = None,
              from_hugging_face: bool = False):
        """
        Build the SSOCAutoCoder with the necessary models and tokenizers.
        """
        # Loading the necessary models and classifiers
        if from_hugging_face:
            self.embedding_model = self._load_embedding_model_from_hugging_face(hugging_face_model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(hugging_face_model_name)
        else:
            self.embedding_model = self._load_embedding_model(embedding_model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
       
        self.full_classifier = self._load_embedding_classifier(full_classifier_path)
        self._check_classifier_validity(self.full_classifier)
        
        self.title_classifier = self._load_embedding_classifier(title_classifier_path)
        self._check_classifier_validity(self.title_classifier)
        
        self.name = model_name

        # After loading the embedding model, set the MAX_TOKEN_LENGTH
        try:
            self.MAX_TOKEN_LENGTH = self.embedding_model.config.max_position_embeddings
        except AttributeError:
            self.MAX_TOKEN_LENGTH = 512

        # Framework identification logic
        if isinstance(self.embedding_model, AutoModel) or isinstance(self.embedding_model, torch.nn.Module):
            self.framework = 'pytorch'
        else:
            raise ValueError("EmbeddingModel should be a Hugging Face transformer model or PyTorch.")

        # Validation for classifier methods
        for classifier in [self.full_classifier, self.title_classifier]:
            if not (callable(getattr(classifier, 'predict', None)) and callable(getattr(classifier, 'predict_proba', None))):
                raise ValueError("Each EmbeddingClassifier must support 'predict' and 'predict_proba' methods.")
            if not hasattr(classifier, 'classes_'):
                raise ValueError("Each EmbeddingClassifier must have the 'classes_' attribute.")

    def _load_embedding_model_from_hugging_face(self, hugging_face_model_name: str):
        """
        Helper method to load the embedding model directly from Hugging Face.
        """
        try:
            model = AutoModel.from_pretrained(hugging_face_model_name)
            return model
        except Exception as e:
            raise ValueError(f"Error loading the embedding model from Hugging Face: {str(e)}")

    def _load_embedding_model(self, embedding_model_path: str):
        """
        Helper method to load the embedding model.
        """
        try:
            model = AutoModel.from_pretrained(embedding_model_path)
            return model
        except Exception as e:
            raise ValueError(f"Error loading the embedding model: {str(e)}")

    def _load_embedding_classifier(self, path_or_obj):
        """
        Helper method to load the embedding classifier.
        """
        try:
            with open(path_or_obj, 'rb') as file:
                return pickle.load(file)
        except ModuleNotFoundError as e:
            missing_module = str(e).split("'")[1]
            raise ImportError(f"Module '{missing_module}' not found. Please ensure you have installed the necessary package to load the EmbeddingClassifier.") from e
        except Exception as e:
            raise ValueError(f"Error loading the embedding classifier: {str(e)}")
        
    def _truncate_text(self, text: str) -> str:
        """
        Truncate the text to fit within the MAX_TOKEN_LENGTH when tokenized.
        """
        tokens = self.tokenizer.tokenize(text)
        if len(tokens) > self.MAX_TOKEN_LENGTH:
            tokens = tokens[:self.MAX_TOKEN_LENGTH]
            truncated_text = self.tokenizer.convert_tokens_to_string(tokens)
            return truncated_text
        return text
                
    def _generate_embeddings(self, text_data: list, task: str, use_gpu: bool = False) -> np.array:
        """
        Generate embeddings for the given text_data using the embedding model.
        
        Args:
        - text_data (list): List of text entries for which embeddings are to be generated.
        - use_gpu (bool): If True, will try to use GPU for generating embeddings.

        Returns:
        - np.array: Array of generated embeddings.
        """
        embeddings = []

        if task == "full":
            for text_pair in text_data:
                title = text_pair[0]
                description = text_pair[1]

                # Truncate text if it exceeds limit
                title = self._truncate_text(title)
                description = self._truncate_text(description)

                # Get embeddings
                embeddings_pair = np.concatenate([self._get_embedding(title, use_gpu), self._get_embedding(description, use_gpu)])
                embeddings.append(embeddings_pair)

        elif task == "title_only":
            for text in text_data:
                # Truncate text if it exceeds limit
                text = self._truncate_text(text)
                # Get embeddings
                embeddings.append(self._get_embedding(text, use_gpu))

        return np.array(embeddings)

    def _get_embedding(self, text: str, use_gpu: bool) -> np.array:
        """
        Generate embedding for a single text entry.

        Args:
        - text (str): Text entry for which embedding is to be generated.
        - use_gpu (bool): If True, will try to use GPU for generating embeddings.

        Returns:
        - np.array: Generated embedding.
        """
        # Tokenization
        text = self.tokenizer(text, return_tensors="pt", padding=True)
        with torch.no_grad():
            if use_gpu and torch.cuda.is_available():
                self.embedding_model.to('cuda')
                text = {key: val.to('cuda') for key, val in text.items()}
                embedding = self.embedding_model(**text).last_hidden_state.mean(dim=1).cpu().numpy().flatten()
            else:
                embedding = self.embedding_model(**text).last_hidden_state.mean(dim=1).numpy().flatten()
        return embedding

    def _get_top_n_predictions(self, embeddings: np.array, classifier, top_n: int) -> dict:
        """
        Retrieve the top N predictions and their probabilities.
        """        
        if embeddings.shape[1] != classifier.n_features_in_:
            raise ValueError(f"Embedding dimensions ({embeddings.shape[1]}) do not match classifier expected input dimensions ({classifier.n_features_in_}).")

        # Extract probabilities
        probabilities = classifier.predict_proba(embeddings)
        top_indices = np.argsort(-probabilities, axis=1)[:, :top_n]
        
        # Convert top predictions and probabilities to Python native types for JSON serialization
        top_predictions = [[classifier.classes_[i].item() if isinstance(classifier.classes_[i], np.integer) else classifier.classes_[i] for i in row] for row in top_indices]
        top_probabilities = [[prob.item() if isinstance(prob, np.floating) else prob for prob in probabilities[j, row]] for j, row in enumerate(top_indices)]        
        
        return {
            "prediction": top_predictions,
            "confidence": top_probabilities
        }
            
    def predict(self, titles: list, descriptions: list, top_n: int = 1, use_gpu: bool = False) -> dict:
        """
        Make predictions using the full_classifier based on titles and descriptions.
        """
        if len(titles) != len(descriptions):
            raise ValueError("The number of titles must match the number of descriptions.")
        text_data = [[title, description] for title, description in zip(titles, descriptions)]
        return self._base_predict(text_data, "full" ,self.full_classifier, top_n, use_gpu)

    def predict_lite(self, titles: list, top_n: int = 1, use_gpu: bool = False) -> dict:
        """
        Make predictions using the title_classifier based on titles.
        """
        return self._base_predict(titles, "title_only", self.title_classifier, top_n, use_gpu)

    def _base_predict(self, text_data: list, task: str, classifier, top_n: int, use_gpu: bool) -> dict:
        """
        Base method to generate embeddings and make predictions.
        """
        embeddings = self._generate_embeddings(text_data, task, use_gpu)
        return self._get_top_n_predictions(embeddings, classifier, top_n)
                
    def save(self, 
             file_name: str):
        """
        Save the SSOCAutoCoder as a pickle file.
        """
        self._pickled_time = str(datetime.now())

        with open(file_name, 'wb') as file:
            pickle.dump(self, file)