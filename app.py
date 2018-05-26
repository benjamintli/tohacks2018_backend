from flask import Flask, request, jsonify
import requests, json
import numpy as np
import keras.preprocessing.text as kpt
from keras.preprocessing.text import Tokenizer
from keras.models import model_from_json


url1 = "https://maps.googleapis.com/maps/api/place/details/json?placeid="
url2 = "&key="
api_key = "AIzaSyDdxOnExyqXGzl36dbk41gbrIZcpRQ037s"


def load_model():
    # initialize flask
    app = Flask(__name__)
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    # and create a model from that
    global model
    model = model_from_json(loaded_model_json)
    # and weight the nodes with saved values
    model.load_weights('model.h5')
    print('model created')
    #return app variable
    return app

app = load_model()

def preprocess_text(input_text):
    tokenizer = Tokenizer(num_words=5000)
    text_array = convert_text_to_index_array(input_text)
    output_text = tokenizer.sequences_to_matrix([text_array], mode='binary')
    return output_text


def convert_text_to_index_array(text):
    with open('dictionary.json', 'r') as dictionary_file:
        dictionary = json.load(dictionary_file)
    words = kpt.text_to_word_sequence(text)
    word_indices = []
    for word in words:
        if word in dictionary:
            word_indices.append(dictionary[word])
    return word_indices


@app.route("/reviews", methods=["POST"])
def get_reviews():
    if request.method == "POST":
        req = request.get_json(force=True)
        url = url1 + req['placeID']+ url2 + api_key
        r = requests.get(url)
        json_dict = r.json()
        return jsonify(json_dict['result']['reviews'])


if __name__ == "__main__":
    app.run()
# @app.route("/predict", methods=["POST"])
# def predict():
