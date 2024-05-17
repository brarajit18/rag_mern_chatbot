import requests

url = 'http://127.0.0.1:5000/ask'
headers = {'Content-Type': 'application/json'}
data = {'question': 'What are pure functions in javascript?'}

response = requests.post(url, headers=headers, json=data)
print(response.json())