import os

from dotenv import load_dotenv
from google import genai


load_dotenv()


class GeminiGenerator:

    def __init__(self):

        print("Initializing Gemini generator...")

        api_key = os.getenv("GOOGLE_API_KEY")

        if not api_key:

            raise RuntimeError(
                "GOOGLE_API_KEY was not found in .env"
            )

        self.client = genai.Client(
            api_key=api_key
        )

        # Use a Gemini model available to your API account.
        self.model_name = os.getenv(
            "GEMINI_MODEL",
            "gemini-2.5-flash"
        )

        print(
            f"Gemini generator initialized: "
            f"{self.model_name}"
        )

    def generate(
        self,
        question,
        context
    ):

        prompt = f"""
You are an IT Infrastructure RAG assistant.

STRICT RULES:

1. Answer ONLY from the provided Context.
2. Do NOT use your own knowledge.
3. Do NOT add information that is not explicitly stated in the Context.
4. If the Context does not contain the answer, respond exactly:

I don't know based on the indexed documentation.

5. Keep the answer concise.
6. Do not invent examples, features, technologies, or details.
7. Do not mention information that is not supported by the Context.

Context:
{context}

Question:
{question}

Answer:
"""

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt
        )

        if not response.text:

            return (
                "I don't know based on the indexed documentation."
            )

        return response.text.strip()