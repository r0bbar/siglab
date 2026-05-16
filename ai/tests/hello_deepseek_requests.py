import requests
import json

# API endpoint (Ollama runs on localhost:11434 by default)
url = 'http://localhost:11434/api/generate'

payload = {
    'model': 'deepseek-r1',
    'prompt': 'Why is the sky blue?',
    'stream': False  # Set to True for token-by-token streaming responses
}

response = requests.post(url, json=payload)

if response.status_code == 200:
    result = response.json()
    print(result['response'])