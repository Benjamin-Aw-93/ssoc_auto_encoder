import pytest
from unittest.mock import patch
import os
import pickle
import numpy as np
from sklearn.dummy import DummyClassifier
from transformers import AutoTokenizer, AutoModel
from ssoc_autocoder.combined_model import SSOCAutoCoder

@pytest.fixture(scope="module")
def dummy_classifier():
    classifier = DummyClassifier(strategy="uniform")
    classifier.fit([[0] * 768], [0])
    yield classifier
    os.remove('dummy_classifier.pkl')

@pytest.fixture
def ssoc_autocoder(dummy_classifier):
    with open('dummy_classifier.pkl', 'wb') as f:
        pickle.dump(dummy_classifier, f)

    ssoc = SSOCAutoCoder()
    yield ssoc

class TestSSOCAutoCoder:
    @patch("ssoc_autocoder.combined_model.AutoTokenizer.from_pretrained")
    @patch("ssoc_autocoder.combined_model.AutoModel.from_pretrained")
    def test_build_from_pretrained(self, mock_model, mock_tokenizer, ssoc_autocoder):
        """ Test building the SSOCAutoCoder using a pre-trained transformer model. """
        mock_tokenizer.return_value = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        mock_model.return_value = AutoModel.from_pretrained("distilbert-base-uncased")
        
        ssoc_autocoder.build(
            model_name="test",
            full_classifier_path="dummy_classifier.pkl",
            title_classifier_path="dummy_classifier.pkl",
            hugging_face_model_name="distilbert-base-uncased",
            from_hugging_face=True
        )
        assert ssoc_autocoder.embedding_model is not None
        assert ssoc_autocoder.tokenizer is not None
        assert ssoc_autocoder.full_classifier is not None
        assert ssoc_autocoder.title_classifier is not None

    def test_predict(self, ssoc_autocoder):
        """ Test making predictions using the full classifier. """
        ssoc_autocoder.build(
            model_name="test",
            full_classifier_path="dummy_classifier.pkl",
            title_classifier_path="dummy_classifier.pkl",
            hugging_face_model_name="distilbert-base-uncased",
            from_hugging_face=True
        )
        with patch("torch.nn.Module.forward", return_value=np.array([[0.5]])):
            predictions = ssoc_autocoder.predict(["software engineer"], ["builds software"], top_n=1)
        assert 'prediction' in predictions
        assert 'confidence' in predictions

    def test_predict_lite(self, ssoc_autocoder):
        """ Test making predictions using the title classifier. """
        ssoc_autocoder.build(
            model_name="test",
            full_classifier_path="dummy_classifier.pkl",
            title_classifier_path="dummy_classifier.pkl",
            hugging_face_model_name="distilbert-base-uncased",
            from_hugging_face=True
        )
        with patch("torch.nn.Module.forward", return_value=np.array([[0.5]])):
            predictions = ssoc_autocoder.predict_lite(["software engineer"], top_n=1)
        assert 'prediction' in predictions
        assert 'confidence' in predictions

    def test_save_and_load(self, ssoc_autocoder):
        """ Test saving and loading the SSOCAutoCoder state. """
        ssoc_autocoder.build(
            model_name="test",
            full_classifier_path="dummy_classifier.pkl",
            title_classifier_path="dummy_classifier.pkl",
            hugging_face_model_name="distilbert-base-uncased",
            from_hugging_face=True
        )
        ssoc_autocoder.save('test_save.pkl')
        assert os.path.exists('test_save.pkl')

        loaded_SSOC = SSOCAutoCoder()
        loaded_SSOC.load('test_save.pkl')
        assert loaded_SSOC.embedding_model is not None
        assert loaded_SSOC.tokenizer is not None
        assert loaded_SSOC.full_classifier is not None
        assert loaded_SSOC.title_classifier is not None

        os.remove('test_save.pkl')