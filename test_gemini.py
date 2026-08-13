from google import genai

from config import GOOGLE_API_KEY

client = genai.Client(api_key=GOOGLE_API_KEY)

response = client.models.generate_content(
    model="gemini-3.1-flash-lite",
    contents="Say hello in one sentence."
)

print(response.text)