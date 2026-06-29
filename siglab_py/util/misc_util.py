import urllib.request

def get_public_ip(
    query_url : str = 'https://api.ipify.org'
):
    with urllib.request.urlopen(query_url) as response:
        public_ip = response.read().decode('utf8')

    return public_ip