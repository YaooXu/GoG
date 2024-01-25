import requests

url = "http://210.75.240.139:18891/label2id"
data = {
    "labels": ["Afghanistan"],
    "question": "Which man is the leader of the country that uses Libya, Libya, Libya as its national anthem?",
}
response = requests.post(url, json=data)

print(response.json())
