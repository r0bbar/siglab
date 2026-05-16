from datetime import datetime
import requests
import ollama

readme_url = "https://raw.githubusercontent.com/r0bbar/siglab/master/siglab_py/README.md"

start_fetch = datetime.now()

try:
    response = requests.get(readme_url)
    response.raise_for_status()
    readme_content = response.text
    
    end_fetch = datetime.now()
    fetch_duration = end_fetch - start_fetch    
    print(f"✅ Successfully fetched from {readme_url}: ({len(readme_content)} characters in {fetch_duration.total_seconds()} sec)")
    
except requests.exceptions.RequestException as e:
    print(f"❌ Error fetching {readme_url}: {e}")
    exit(1)

prompt = f"""
Please have a look at:
{readme_content}

Please provide a concise summary, in tabular format.
"""

start_think = datetime.now()
response = ollama.chat(
    model='deepseek-r1',
    messages=[{
        'role': 'user',
        'content': prompt,
    }]
)
end_think = datetime.now()
think_duration = end_think - start_think

print(f"\n--- DeepSeek's Response ({think_duration.total_seconds()}) ---\n")
print(response['message']['content'])