import json
from config import settings
from google import genai

client = genai.Client(api_key=settings.gemini_api_key)

print("Listing available Gemini models for your API key:")
try:
    models = list(client.models.list())
    for m in models:
        # Filter for generateContent supported models
        name = getattr(m, 'name', '') or str(m)
        print(" -", name)
except Exception as e:
    print("Error listing models:", e)

