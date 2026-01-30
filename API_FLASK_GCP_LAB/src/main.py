from flask import Flask, request, jsonify
from predict import predict_diabetes
import os

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    age = float(data['age'])
    sex = float(data['sex'])
    bmi = float(data['bmi'])
    bp = float(data['bp'])
    s1 = float(data['s1'])
    s2 = float(data['s2'])
    s3 = float(data['s3'])
    s4 = float(data['s4'])
    s5 = float(data['s5'])
    s6 = float(data['s6'])

    prediction = predict_diabetes(age, sex, bmi, bp, s1, s2, s3, s4, s5, s6)

    return jsonify({'prediction': round(float(prediction), 2)})

if __name__ == '__main__':
    app.run(
        debug=True,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8080))
    )