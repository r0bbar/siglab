import ollama

response = ollama.generate(
    model='deepseek-r1',
    prompt='Why is the sky blue?'
)
print(response['response'])

response = ollama.chat(
    model='deepseek-r1',
    messages=[{
        'role': 'user',
        'content': 'Why is the sky blue?',
    }]
)
print(response['message']['content'])