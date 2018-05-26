from flask import Flask, request, jsonify
import requests, json

app = Flask(__name__)

url1 = "https://maps.googleapis.com/maps/api/place/details/json?placeid="
url2 = "&key="
api_key = "AIzaSyDdxOnExyqXGzl36dbk41gbrIZcpRQ037s"


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
