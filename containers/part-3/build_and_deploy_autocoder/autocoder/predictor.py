# This is the file that implements a flask server to do inferences. 
from __future__ import print_function

local_testing = True

import io
import json
import os
import pickle
import signal
import sys
import traceback

import flask
from ssoc_autocoder.combined_model import SSOCAutoCoder

if local_testing:
    model_path = "/Users/dsaid/Desktop/mom-projects/ssoc_autocoder/Models/"
else:
    prefix = "/opt/ml/"
    model_path = os.path.join(prefix, "model")

# The flask app for serving predictions
app = flask.Flask(__name__)

# A singleton for holding the model. This simply loads the model and holds it.
# It has a predict function that does a prediction based on the model and the input data.

class ScoringService(object):
    model = None  # Where we keep the model when it's loaded

    @classmethod
    def get_model(cls):
        """Get the model object for this instance, loading it if it's not already loaded."""
        if cls.model == None:
            with open(os.path.join(model_path, "autocoder.pkl"), "rb") as inp:
                cls.model = pickle.load(inp)
        return cls.model

    @classmethod
    def predict(cls, task, text, top_n):
        """For the input, do the predictions and return them.

        Args:
            input (a pandas dataframe): The data on which to do the predictions. There will be
                one prediction per row in the dataframe"""
        clf = cls.get_model()

        if task == "title-only":
            print(text)
            return clf.predict_lite(title=text[0], top_n=top_n)
        else:
            return clf.predict(title=text[0], description=text[1], top_n=top_n)

@app.route("/ping", methods=["GET"])
def ping():
    """Determine if the container is working and healthy. In this container, we declare
    it healthy if we can load the model successfully.
    """
    health = ScoringService.get_model() is not None  # You can insert a health check here

    status = 200 if health else 404
    return flask.Response(response="\n", status=status, mimetype="application/json")


@app.route("/predictSSOC", methods=["POST"])
def transformation():
    """
    Do an inference on a single batch of data. In this sample server, we take data as JSON, extract
    the necessary fields, and then convert the predictions back to JSON.

    Returns:
        flask.Response: A Flask response object containing the predictions in JSON format.
    """
    data = None

    # Convert from JSON to pandas
    if flask.request.content_type == "application/json":
        input_json = flask.request.get_json()
        task = str(input_json['task'])
        job_title = str(input_json['job_title'])
        job_description = str(input_json['job_description'])
        top_n = input_json['top_n']
        if task != "title-only" and task != "full":
            return flask.Response(
                response="Invalid task value. Task should be either 'title-only' or 'full'", status=400, mimetype="text/plain"
            )
        if not isinstance(top_n, int) or top_n < 1 or top_n > 10:
            return flask.Response(
                response="Invalid top_n value. top_n should be an integer between 1 and 10", status=400, mimetype="text/plain"
            )
    else:
        return flask.Response(
            response="This predictor only supports JSON data", status=415, mimetype="text/plain"
        )

    # Do the prediction
    predictions = ScoringService.predict(task, [job_title, job_description], top_n)
    predictions_json = json.dumps(predictions)

    return flask.Response(response=predictions_json, status=200, mimetype="application/json")

if local_testing:
    if __name__ == "__main__":
        app.run(host='0.0.0.0', port=8080, debug=True)