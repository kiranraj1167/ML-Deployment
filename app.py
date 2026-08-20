from flask import Flask, request
import pickle

app = Flask(__name__)

@app.route('/prediction', methods=['POST'])
def preds():
    sl=float(request.form['sl'])
    sw=float(request.form['sw'])
    pl=float(request.form['pl'])
    pw=float(request.form['pw'])

    with open('iris-model.pkl', 'rb') as f:
        model =pickle.load(f)
    prediction_result = model.predict([[sl,sw,pl,pw]])

    return f"Prediction value is {prediction_result}"
app.run()