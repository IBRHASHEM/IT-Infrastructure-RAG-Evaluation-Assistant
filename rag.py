import time

from embeddings import EmbeddingGenerator
from vector_store import VectorStore

from google import genai
from google.genai import types
from google.genai import errors

from config import GOOGLE_API_KEY, MODEL_NAME
print("Sending request to Gemini...")

class RAG:

    def __init__(self):
        self.embedder = EmbeddingGenerator()
        self.vector_db = VectorStore()
        self.client = genai.Client(api_key=GOOGLE_API_KEY)

    def search(self, question: str, k: int = 5):

        query_embedding = self.embedder.embed(question)

        return self.vector_db.search(
            query_embedding,
            k=k
        )

    def ask(self, question: str):

        results = self.search(question, k=5)

        # -----------------------------
        # No search results
        # -----------------------------
        if not results["documents"] or len(results["documents"][0]) == 0:
            return {
                "answer": "I don't know based on the indexed documentation.",
                "sources": []
            }

        print("=" * 80)
        
        print("=" * 80)
        for i, meta in enumerate(results["metadatas"][0], start=1):
            print(
                f"{i}. {meta['source']} "
                f"(Page {meta['page']}) "
                f"Distance={results['distances'][0][i-1]:.3f}"
            )

        context = "\n\n".join(results["documents"][0])

        prompt = f"""
                You are an expert IT Infrastructure Engineer.

                Instructions:
                - Answer ONLY using the Context.
                - Never use outside knowledge.
                - If information is incomplete say:
                "I don't know based on the indexed documentation."
                - Write concise technical answers.
                - Use bullet points when appropriate.

Context:
{context}

Question:
{question}

Answer:
"""

        
        print("=" * 80)
        print("=" * 80)
        print("Prompt Preview")
        print("=" * 80)
        print(prompt[:1000] + "\n...")
        # -----------------------------
        # Retry
        # -----------------------------
        max_retries = 3

        for attempt in range(max_retries):

            try:

                response = self.client.models.generate_content(
                    model=MODEL_NAME,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.2
                    )
                )

                break

            except errors.ServerError as e:

                print("Gemini Server Error:")
                print(e)

                if attempt == max_retries - 1:
                    return {
                        "answer": "Google Gemini server is temporarily unavailable.",
                        "sources": []
                    }

                print("Retrying in 5 seconds...")
                time.sleep(5)

            except errors.ClientError as e:

                print("Gemini Client Error:")
                print(e)

                return {
                    "answer": f"Gemini Client Error: {e}",
                    "sources": []
                }

            except Exception as e:

                print("Unexpected Error:", e)

                return {
                    "answer": f"Unexpected Error: {e}",
                    "sources": []
                }

        # -----------------------------
        # Sources
        # -----------------------------
        sources = []

        for meta in results["metadatas"][0]:

            item = {
                "source": meta["source"],
                "page": meta["page"]
            }

            if item not in sources:
                sources.append(item)

        return {
            "answer": response.text,
            "sources": sources
        }