from flask import Flask,request,jsonify
import numpy as np
import pickle
model = pickle.load(open('heart1.pkl', 'rb'))
app = Flask(__name__)
@app.route('/')
def index():
    return "Hello world"
@app.route('/predict',methods=['POST'])
def predict():
    cp = request.form.get('cp')
    thalachh = request.form.get('thalachh')
    slp = request.form.get('slp')
    restecg = request.form.get('restecg')
    chol = request.form.get('chol')
    trtbps = request.form.get('trtbps')
    fbs = request.form.get('fbs')
    oldpeak = request.form.get('oldpeak')
    input_query = np.array([[cp, thalachh, slp, restecg, chol, trtbps, fbs, oldpeak]])
    result = model.predict(input_query)[0]
    return jsonify({'hearth_disease':str(result)})
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000,debug=True)