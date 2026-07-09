import requests

response = requests.post(
    "http://localhost:8000/api/speak", 
    json={"words": ["Hello", "Help", "Water"]}
)
print("Status Code:", response.status_code)
if response.status_code != 200:
    print("Response text:", response.text)
else:
    print("Success. Audio content length:", len(response.content))
