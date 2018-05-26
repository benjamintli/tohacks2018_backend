from flask import Flask, request, jsonify
import requests, json
import numpy as np
import keras.preprocessing.text as kpt
from keras.preprocessing.text import Tokenizer
from keras.models import model_from_json
import tensorflow as tf

model = None
tokenizer = Tokenizer(num_words=5000)

url1 = "https://maps.googleapis.com/maps/api/place/details/json?placeid="
url2 = "&key="
api_key = "AIzaSyDdxOnExyqXGzl36dbk41gbrIZcpRQ037s"
labels = ['fake', 'real']

app = Flask(__name__)

@app.route("/reviews", methods=["POST"])
def get_reviews():
    if request.method == "POST":
        req = request.get_json(force=True)
        url = url1 + req['placeID'] + url2 + api_key
        r = requests.get(url)
        json_dict = r.json()
        return jsonify(json_dict)

@app.route("/predict", methods=["POST"])
def get_post():
    tokenizer = Tokenizer(num_words=5000)
    # for human-friendly printing
    labels = ['fake', 'real']

    # read in our saved dictionary
    with open('training_data/dictionary.json', 'r') as dictionary_file:
        dictionary = json.load(dictionary_file)

    def convert_text_to_index_array(text):
        words = kpt.text_to_word_sequence(text)
        wordIndices = []
        for word in words:
            if word in dictionary:
                wordIndices.append(dictionary[word])
        return wordIndices

    # read in your saved model structure
    json_file = open('training_data/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    # and create a model from that
    model = model_from_json(loaded_model_json)
    # and weight your nodes with your saved values
    model.load_weights('training_data/model.h5')

    req = request.get_json(force=True)
    url = url1 + req['placeID'] + url2 + api_key
    r = requests.get(url)
    json_dict = r.json()
    if 'reviews' in json_dict['result']:
        for i in range(0, 5):
            testArr = convert_text_to_index_array(json_dict['result']['reviews'][i]['text'])
            input_eval = tokenizer.sequences_to_matrix([testArr], mode='binary')
            # predict which bucket your input belongs in
            pred = model.predict(input_eval)
            pred_accuracy = {"accuracy": (100*float(pred[0][1]))}
            pred_value = {"polarity" : labels[np.argmax(pred)]}
            json_dict['result']['reviews'][i].update(pred_accuracy)
        return jsonify(json_dict['result']['reviews'])
    else:
        return jsonify({"text": "no reviews exist"})

if __name__ == "__main__":
    app.run()
