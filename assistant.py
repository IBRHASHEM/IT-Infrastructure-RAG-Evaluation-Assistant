# from google import genai
# from google.genai import types

# from config import GEMINI_API_KEY, MODEL_NAME


# class GeminiAssistant:

#     def __init__(self):
#         self.client = genai.Client(api_key=GEMINI_API_KEY)

#     def ask(self, question: str):

#         response = self.client.models.generate_content(
#             model=MODEL_NAME,
#             contents=question,
#             config=types.GenerateContentConfig(
#                 temperature=0.2
#             ),
#         )

#         return response.text
from rag import RAG


class GeminiAssistant:

    def __init__(self):
        self.rag = RAG()

    def ask(self, question: str):
        return self.rag.ask(question)