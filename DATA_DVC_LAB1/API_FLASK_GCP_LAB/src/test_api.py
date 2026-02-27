import requests
import json

url = 'http://127.0.0.1:8080/predict'

payload = {
    'age': 0.038,
    'sex': 0.051,
    'bmi': 0.062,
    'bp': 0.022,
    's1': -0.044,
    's2': -0.035,
    's3': -0.043,
    's4': -0.003,
    's5': 0.019,
    's6': -0.017
}

headers = {
    'Content-Type': 'application/json'
}

response = requests.post(url, data=json.dumps(payload), headers=headers)

print("Status:", response.status_code)
print("Body:", response.text)

if response.status_code == 200:
    try:
        prediction = response.json()['prediction']
        print('Predicted disease progression:', prediction)
    except Exception as e:
        print("Could not parse JSON:", e)
else:
    print('Error:', response.status_code)