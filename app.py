import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle

app = Flask(__name__, template_folder = "templates", static_folder="static")
model = pickle.load(open('heart_disease_prediction.pickle', 'rb'))

@app.route('/')
def home():
    if request.method=='POST':
        return 'POST'
    return render_template('index.html')

@app.route('/predict',methods=['POST','GET'])
def predict():
        input_features = [float(x) for x in request.form.values()]
        features_value = [np.array(input_features)]
        features_name = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal']
        df = pd.DataFrame(features_value)
        output = model.predict(df)
        if output == 0:
             res_val = "Healthy heart"
        else:
            res_val = "Defective heart"
        return render_template('index.html', prediction_text='Patient has {}'.format(res_val))

if __name__ == "__main__":
    app.run()