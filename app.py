from flask import Flask,request,jsonify
import numpy as np
import pickle
model = pickle.load(open('model5.pkl', 'rb'))
app = Flask(__name__)
@app.route('/')
def index():
    return "Hello world"
@app.route('/predict',methods=['POST'])
def predict():
    concave_points_mean= request.form.get('concave_points_mean')
    area_mean = request.form.get('area_mean')
    radius_mean = request.form.get('radius_mean')
    perimeter_mean = request.form.get('perimeter_mean')
    concavity_mean = request.form.get('concavity_mean')
    input_query = np.array([[concave_points_mean,area_mean,radius_mean,perimeter_mean,concavity_mean]])
    result = model.predict(input_query)[0]
    return jsonify({'diagnosis':str(result)})


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)