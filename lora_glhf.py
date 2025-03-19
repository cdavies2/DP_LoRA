import os, pytest, json
from openai import OpenAI

# Connect to the GLHF API

client1 = OpenAI(
    api_key=os.environ.get("GLHF_API_KEY"),
    base_url="https://glhf.chat/api/openai/v1",
)
