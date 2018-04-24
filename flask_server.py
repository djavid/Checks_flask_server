from flask import Flask, request
import Core

app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/predict', methods=['POST'])
def predict():
    content = request.get_json()
    values = content.get('values')
    return Core.predict_proba(values)


if __name__ == '__main__':
    app.run()
