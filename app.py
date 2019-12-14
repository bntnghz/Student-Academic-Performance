import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/form_predict')
def form_predict():
    return render_template('predict.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    final_features = [np.array(features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    if output == 0:
        out = 'High'
    elif output == 1:
        out = 'Low'
    else:
        out = 'Medium'

    return render_template('result_predict.html', prediction_text='{}'.format(out))

if __name__ == "__main__":
    app.run(debug=True)