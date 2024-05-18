import requests

url = 'http://18.206.219.103/ask'
headers = {'Content-Type': 'application/json'}
data = {'question': 'What are pure functions in javascript?'}

response = requests.post(url, headers=headers, json=data)
print(response.json())