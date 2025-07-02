import os
from Constants import ENV_FILE
from dotenv import load_dotenv, set_key

load_dotenv(ENV_FILE)

def load_api_key():
        return os.getenv("OPENAI_API_KEY", "")
    
def save_api_key(key):
    set_key(ENV_FILE, "OPENAI_API_KEY", key)