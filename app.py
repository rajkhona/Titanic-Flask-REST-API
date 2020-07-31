import joblib
from flask import Flask, request, jsonify, render_template
from model import filename1, transform_prediction_data
import numpy as np
from joblib import dump, load

app = Flask(__name__)
loaded_model = joblib.load(filename1)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    final_transformed = transform_prediction_data(final_features)
    prediction = loaded_model.predict(final_transformed)
    prediction_proba = loaded_model.predict_proba(final_transformed)
    if prediction == 1:
        return render_template('index.html',
                               prediction_text='Survived and your '
                                               'probability of survival is: {}'.format(prediction_proba))
    else:
        return render_template('index.html',
                               prediction_text='Sorry did not survive '
                                               'probability of not surviving is: {}'.format(prediction_proba))


if __name__ == '__main__':
    app.run()
