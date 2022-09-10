import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import os

#app name
app = Flask(__name__)


#Load the saved model
def load_model():
    return pickle.load(open('iris-model.pkl','rb'))

#homepage
@app.route('/')
def home():
    return render_template('index.html')

#Predict the result and return it
@app.route('/predict',methods=['POST'])
def predict():

    labels = ['setosa', 'versicolor', 'virginica']

    features = [float(x) for x in request.form.values()]
    print(features)

    values = [np.array(features)]

    print(values)
    model = load_model()
    prediction = model.predict(values)

    result = labels[prediction[0]]

    return render_template('index.html', output='The flower is {}'.format(result))


if __name__ == "__main__":
    port=int(os.environ.get('PORT', 5000))
    app.run(debug=True, use_reloader=True)

